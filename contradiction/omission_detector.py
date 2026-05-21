"""
contradiction/omission_detector.py
------------------------------------
Detects omission contradictions: topics that an executive consistently mentioned
across 3+ consecutive prior quarters but then completely dropped in the current quarter.

Logic:
  1. Extract noun-phrase "topics" from each statement using spaCy.
  2. Build a per-executive topic frequency map across chronological quarters.
  3. For each quarter Q_t, check if a topic appeared in Q_{t-3}, Q_{t-2}, Q_{t-1}
     (all three consecutive) but is completely absent in Q_t.
  4. Flag each such dropout as an OMISSION contradiction.

The omission is stored in the contradictions table as:
  - statement_a_id: the most recent prior statement mentioning the topic
  - statement_b_id: the first statement the executive made in Q_t (the "anchor")
  - contradiction_type: "OMISSION"
  - score: 1.0 (binary — either the topic was dropped or it wasn't)
  - details: JSON with topic, prior_quarters, omitted_quarter
"""

import sys
import re
import spacy
from pathlib import Path
from loguru import logger
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import OMISSION_MIN_PRIOR_QUARTERS, SPACY_MODEL

# ─────────────────────────────────────────────────────────────────────────────
# spaCy model (singleton)
# ─────────────────────────────────────────────────────────────────────────────
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        logger.info(f"Loading spaCy model: {SPACY_MODEL}...")
        _nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
        _nlp.enable_pipe("senter")   # sentence segmentation only
        logger.info("spaCy model loaded.")
    return _nlp


# ─────────────────────────────────────────────────────────────────────────────
# Stop words / noise to filter out of noun chunks
# ─────────────────────────────────────────────────────────────────────────────
_STOP_TOKENS = {
    "we", "our", "i", "you", "they", "it", "this", "that", "these", "those",
    "the", "a", "an", "which", "who", "what", "quarter", "year", "time",
    "basis", "terms", "way", "level", "kind", "part", "lot", "things",
    "thing", "areas", "area", "side", "focus", "view"
}

# Minimum characters for a topic to be meaningful
_MIN_TOPIC_LEN = 4

# Maximum words in a noun phrase to remain tractable
_MAX_PHRASE_WORDS = 4


def _extract_topics(text: str) -> List[str]:
    """
    Extract cleaned noun phrases from text.
    Returns a deduplicated list of lowercase topic strings.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    topics = set()
    for chunk in doc.noun_chunks:
        # Remove leading determiners/pronouns — take the root+right tokens
        tokens = [
            t.text.lower() for t in chunk
            if t.pos_ not in ("DET", "PRON") and t.text.lower() not in _STOP_TOKENS
        ]
        # Filter on length
        phrase = " ".join(tokens).strip()
        if (
            len(phrase) >= _MIN_TOPIC_LEN
            and len(tokens) <= _MAX_PHRASE_WORDS
            and not phrase.isdigit()
        ):
            topics.add(phrase)
    return list(topics)


# ─────────────────────────────────────────────────────────────────────────────
# Quarter ordering helper
# ─────────────────────────────────────────────────────────────────────────────

# Canonical quarter sort key: Q1FY23 → (2023, 1)
_QUARTER_RE = re.compile(r"Q(\d)FY(\d{2,4})", re.IGNORECASE)


def _quarter_sort_key(quarter: str) -> Tuple[int, int]:
    m = _QUARTER_RE.match(quarter)
    if not m:
        return (9999, 9)
    q_num = int(m.group(1))
    fy    = int(m.group(2))
    if fy < 100:
        fy += 2000   # FY23 → 2023
    return (fy, q_num)


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

def detect_omissions(
    statements: List[Dict[str, Any]],
    min_prior_quarters: int = OMISSION_MIN_PRIOR_QUARTERS,
) -> List[Dict[str, Any]]:
    """
    Given a list of statement dicts for a SINGLE executive (sorted or unsorted),
    detect omission contradictions.

    Args:
        statements: List of statement dicts, each must have:
                    'id', 'text', 'quarter', 'year'
        min_prior_quarters: How many consecutive prior quarters a topic must
                            appear in before its absence becomes significant.

    Returns:
        List of omission event dicts, each with:
          - statement_a_id      : int  (most recent prior mention of the topic)
          - statement_b_id      : int  (first statement in the omitted quarter)
          - contradiction_type  : "OMISSION"
          - score               : 1.0
          - details             : dict with topic, prior_quarters, omitted_quarter
    """
    if not statements:
        return []

    # Group statements by quarter and sort quarters chronologically
    by_quarter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for stmt in statements:
        by_quarter[stmt["quarter"]].append(stmt)

    sorted_quarters = sorted(by_quarter.keys(), key=_quarter_sort_key)

    if len(sorted_quarters) < min_prior_quarters + 1:
        # Not enough quarters to detect omissions
        return []

    # Build topic → list of (quarter, statement_id) occurrences
    # topic_occurrences[topic][quarter] = list of statement IDs
    topic_occurrences: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for quarter, stmts in by_quarter.items():
        for stmt in stmts:
            topics = _extract_topics(stmt.get("text", ""))
            for topic in topics:
                topic_occurrences[topic][quarter].append(stmt["id"])

    # Find omissions
    omissions = []
    seen_pairs = set()   # Deduplicate same topic+omitted_quarter combos

    for topic, quarter_map in topic_occurrences.items():
        for i, current_q in enumerate(sorted_quarters):
            if i < min_prior_quarters:
                continue   # Not enough history before this quarter

            # Get the N consecutive prior quarters
            prior_quarters = sorted_quarters[i - min_prior_quarters : i]

            # Check: topic must appear in ALL prior quarters
            topic_in_all_prior = all(q in quarter_map for q in prior_quarters)
            if not topic_in_all_prior:
                continue

            # Check: topic must be ABSENT in current quarter
            # (current quarter must also have some statements to confirm it's
            # an active quarter for this executive, not just missing data)
            if current_q in quarter_map:
                continue   # Topic mentioned in current quarter — no omission

            if not by_quarter.get(current_q):
                continue   # Executive has no statements in current quarter at all

            # Deduplicate
            dedup_key = (topic, current_q)
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)

            # Find the most recent prior statement mentioning the topic
            most_recent_prior_q = prior_quarters[-1]
            stmt_a_id = quarter_map[most_recent_prior_q][-1]   # last statement in that quarter

            # Anchor statement_b to the first statement in the omitted quarter
            stmt_b_id = by_quarter[current_q][0]["id"]

            omissions.append({
                "statement_a_id":     stmt_a_id,
                "statement_b_id":     stmt_b_id,
                "contradiction_type": "OMISSION",
                "score":              1.0,
                "details": {
                    "topic":           topic,
                    "prior_quarters":  prior_quarters,
                    "omitted_quarter": current_q,
                },
            })

    return omissions
