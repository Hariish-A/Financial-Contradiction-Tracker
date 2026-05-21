"""
contradiction/soft_detector.py
-------------------------------
Detects soft contradictions between two statements using three composite signals:

  1. Topic Similarity     — cosine similarity of statement embeddings (0.0 → 1.0)
  2. Sentiment Flip       — FinBERT labels from DB (positive ↔ negative = 1.0, etc.)
  3. Hedge Escalation     — HEDGE_SCALE keyword delta (was confident, now cautious = high score)

Composite formula (from note.md):
  score = 0.4 * topic_similarity + 0.4 * sentiment_flip + 0.2 * hedge_escalation

Threshold: score > SOFT_CONTRADICTION_THRESHOLD (default 0.6) → SOFT contradiction
"""

import sys
import re
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import HEDGE_SCALE, SOFT_CONTRADICTION_THRESHOLD
from storage.database import load_embedding


# ─────────────────────────────────────────────────────────────────────────────
# 1. Topic Similarity
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised vectors.
    Since embeddings are already normalised, this is just the dot product.
    """
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(emb_a / norm_a, emb_b / norm_b))


def _topic_similarity(stmt_a: Dict[str, Any], stmt_b: Dict[str, Any]) -> float:
    """
    Compute cosine similarity between the embeddings of two statements.
    Returns 0.0 if either statement has no embedding.
    """
    blob_a = stmt_a.get("embedding")
    blob_b = stmt_b.get("embedding")
    if blob_a is None or blob_b is None:
        return 0.0
    try:
        emb_a = load_embedding(blob_a)
        emb_b = load_embedding(blob_b)
        return _cosine_similarity(emb_a, emb_b)
    except Exception as e:
        logger.warning(f"Could not compute topic similarity: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sentiment Flip
# ─────────────────────────────────────────────────────────────────────────────

# Canonical sentiment groups
_POSITIVE = {"positive"}
_NEGATIVE = {"negative"}
_NEUTRAL  = {"neutral"}


def _sentiment_flip_score(sentiment_a: Optional[str], sentiment_b: Optional[str]) -> float:
    """
    Returns a score representing how much the sentiment has flipped between two statements.
      - positive ↔ negative (hard flip): 1.0
      - positive ↔ neutral or negative ↔ neutral (soft flip): 0.5
      - same sentiment group: 0.0
    """
    a = (sentiment_a or "neutral").lower().strip()
    b = (sentiment_b or "neutral").lower().strip()

    if a == b:
        return 0.0

    # Hard flip: positive ↔ negative
    if (a in _POSITIVE and b in _NEGATIVE) or (a in _NEGATIVE and b in _POSITIVE):
        return 1.0

    # Soft flip: one neutral, one non-neutral
    if (a in _NEUTRAL) != (b in _NEUTRAL):
        return 0.5

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hedge Escalation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_hedge_score(text: str) -> float:
    """
    Scan text for HEDGE_SCALE keywords (whole-word, case-insensitive).
    Returns the minimum matched hedge score (i.e. most cautious keyword wins).
    If no keywords match, returns 1.0 (default = confident/unhedged).
    """
    text_lower = text.lower()
    matched_scores = []
    for keyword, score in HEDGE_SCALE.items():
        # Whole-word match to avoid partial hits (e.g. "strong" inside "strengthen")
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, text_lower):
            matched_scores.append(score)
    if not matched_scores:
        return 1.0   # No hedge keywords → confident/unhedged
    return min(matched_scores)


def _hedge_escalation_score(text_a: str, text_b: str) -> float:
    """
    Measures how much more negative/cautious statement B is compared to A.
    Positive score means B is more hedged than A (escalation = contradiction signal).
    Result clamped to [0.0, 1.0].
    """
    score_a = _extract_hedge_score(text_a)
    score_b = _extract_hedge_score(text_b)
    # If A was more confident and B became more cautious → escalation
    escalation = score_a - score_b
    return float(max(0.0, escalation))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Composite Scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_soft_contradiction(
    stmt_a: Dict[str, Any],
    stmt_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute a composite soft-contradiction score between two statement dicts.

    Each dict must contain at minimum:
      - 'text'      : str
      - 'sentiment' : str (positive | negative | neutral)
      - 'embedding' : bytes | None (serialised numpy float32 blob)

    Returns a dict with:
      - topic_similarity   : float [0, 1]
      - sentiment_flip     : float {0, 0.5, 1}
      - hedge_escalation   : float [0, 1]
      - composite_score    : float [0, 1]
      - is_soft_contradiction : bool
    """
    text_a = stmt_a.get("text", "")
    text_b = stmt_b.get("text", "")

    topic_sim   = _topic_similarity(stmt_a, stmt_b)
    sent_flip   = _sentiment_flip_score(stmt_a.get("sentiment"), stmt_b.get("sentiment"))
    hedge_esc   = _hedge_escalation_score(text_a, text_b)

    composite = (
        0.4 * topic_sim
        + 0.4 * sent_flip
        + 0.2 * hedge_esc
    )
    composite = round(float(composite), 6)

    return {
        "topic_similarity":      round(topic_sim, 4),
        "sentiment_flip":        round(sent_flip, 4),
        "hedge_escalation":      round(hedge_esc, 4),
        "composite_score":       composite,
        "is_soft_contradiction": composite > SOFT_CONTRADICTION_THRESHOLD,
    }
