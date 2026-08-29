"""
credibility/scorer.py
---------------------
Milestone 5 — Executive Credibility Scorer.

Two complementary scoring components:

1. PredictionExtractor
   - Scans QUANTITATIVE_GUIDANCE statements using regex patterns to pull out
     numeric predictions (%, INR values, BPS, etc.) and store them in the
     `predictions` table with a direction tag (up / down / stable).

2. CredibilityScorer
   - For each executive, pulls:
       a) All their contradiction records (HARD / SOFT / OMISSION)
       b) All their verified predictions (actual_value populated)
   - Computes a composite credibility score [0–100]:

       base        = 100
       - hard_count   × PENALTY_HARD        (default -20)
       - soft_count   × PENALTY_SOFT        (default -10)
       - omit_count   × PENALTY_OMISSION    (default  -5)
       + dir_correct  × REWARD_CORRECT      (default +10)
       - dir_wrong    × PENALTY_WRONG       (default -10)

       clamped to [0, 100]

   - Additionally computes direction_accuracy and avg_magnitude_error_pct
     for verified predictions (as per note.md formula).

Usage:
    from credibility.scorer import PredictionExtractor, CredibilityScorer

    extractor = PredictionExtractor()
    inserted  = extractor.run()          # populates predictions table

    scorer    = CredibilityScorer()
    report    = scorer.score_all()       # returns list of result dicts
    single    = scorer.score_executive(exec_id=1)
"""

import re
import sys
import json
from pathlib import Path
from loguru import logger
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from storage.database import (
    get_connection,
    get_all_executives,
    get_statements_for_executive,
    get_predictions_for_executive,
    get_contradictions_for_executive,
    get_statement_count_for_executive,
    insert_prediction,
    DB_PATH,
)

# ─────────────────────────────────────────────────────────────────────────────
# Scoring weights (easy to tune in one place)
# ─────────────────────────────────────────────────────────────────────────────
PENALTY_HARD        = 20    # deducted per HARD contradiction
PENALTY_SOFT        = 10    # deducted per SOFT contradiction
PENALTY_OMISSION    = 5     # deducted per OMISSION contradiction
REWARD_CORRECT      = 10    # added per correctly-called direction
PENALTY_WRONG       = 10    # deducted per incorrectly-called direction


# ─────────────────────────────────────────────────────────────────────────────
# Metric keyword → canonical name mapping
# ─────────────────────────────────────────────────────────────────────────────
_METRIC_MAP = {
    # revenue / top-line
    "revenue":         "revenue_growth",
    "top-line":        "revenue_growth",
    "topline":         "revenue_growth",
    "sales":           "revenue_growth",
    # margin / profitability
    "margin":          "margin",
    "ebitda":          "ebitda_margin",
    "operating margin":"operating_margin",
    "net margin":      "net_margin",
    # growth
    "growth":          "revenue_growth",
    "advance growth":  "loan_growth",
    "loan growth":     "loan_growth",
    "credit growth":   "loan_growth",
    # returns
    "roe":             "roe",
    "return on equity":"roe",
    "roi":             "roi",
    "return on investment": "roi",
    "eps":             "eps",
    # capex
    "capex":           "capex",
    "capital expenditure": "capex",
    # headcount / hiring
    "headcount":       "headcount",
    "hiring":          "headcount",
    # deposit / nii
    "deposit":         "deposit_growth",
    "nii":             "nii_growth",
    "net interest":    "nii_growth",
}


def _map_metric(text_window: str) -> str:
    """Map a short text window around a number to a canonical metric name."""
    window = text_window.lower()
    for kw, canonical in _METRIC_MAP.items():
        if kw in window:
            return canonical
    return "other"


def _infer_direction(text_window: str, value: float) -> str:
    """
    Infer the predicted direction from surrounding context words.
    Returns 'up', 'down', or 'stable'.
    """
    up_words    = {"grow", "growth", "increase", "expand", "higher", "rise",
                   "improve", "target", "expect", "positive", "robust", "strong"}
    down_words  = {"decline", "fall", "reduce", "decrease", "lower", "headwind",
                   "suppress", "compress", "pressure", "slow", "contract"}
    stable_words= {"stable", "steady", "flat", "maintain", "sustain", "hold"}

    tokens = set(text_window.lower().split())
    if tokens & stable_words:
        return "stable"
    if tokens & down_words:
        return "down"
    return "up"   # default for forward-guidance numbers is usually bullish


# ─────────────────────────────────────────────────────────────────────────────
# 1. PredictionExtractor
# ─────────────────────────────────────────────────────────────────────────────

# Regex patterns, each returns one captured numeric group
_PREDICTION_PATTERNS = [
    # "18% growth / margin / revenue"
    re.compile(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:to\s+\d+(?:\.\d+)?\s*%\s*)?(?:growth|revenue|margin|roe|roi|eps|increase|expansion|target|guidance|improvement|return|yield)",
        re.IGNORECASE,
    ),
    # "grow / expand by 18%"
    re.compile(
        r"(?:grow|expand|increase|improve|rise|target|expect)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    # "revenue growth of 18%"
    re.compile(
        r"(?:revenue|margin|growth|roe|eps|return|capex|deposit|nii)\s+(?:growth\s+)?(?:of|at|around|approximately|~)\s*(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    # "INR / Rs 400 billion/crore"
    re.compile(
        r"(?:INR|Rs\.?|₹)\s*(\d+(?:\.\d+)?)\s*(?:billion|bn|crore|cr|lakh|million)",
        re.IGNORECASE,
    ),
    # basis points: "200 bps improvement"
    re.compile(
        r"(\d+)\s*(?:bps|basis\s+points?)\s*(?:improvement|expansion|decline|reduction|compression)",
        re.IGNORECASE,
    ),
]

# Only extract from guidance-type statements
_GUIDANCE_TYPES = {
    "QUANTITATIVE_GUIDANCE",
    "QUALITATIVE_GUIDANCE",
    "HEDGED",
}

_WINDOW_CHARS = 120   # characters around the match to use for metric/direction inference


class PredictionExtractor:
    """
    Scans statements of type QUANTITATIVE_GUIDANCE (and related types) for numeric
    predictions and populates the `predictions` table.
    """

    def run(self, executive_id: Optional[int] = None) -> int:
        """
        Extract predictions for all executives (or one if executive_id is set).
        Returns total number of predictions inserted.
        """
        conn = get_connection()
        if executive_id is not None:
            executives = conn.execute(
                "SELECT id, name, role FROM executives WHERE id=?", (executive_id,)
            ).fetchall()
        else:
            executives = conn.execute(
                "SELECT id, name, role FROM executives ORDER BY id"
            ).fetchall()
        conn.close()

        # Pre-load existing (executive_id, statement_id) pairs to avoid dupes
        conn = get_connection()
        existing = set(
            conn.execute(
                "SELECT executive_id, statement_id FROM predictions"
            ).fetchall()
        )
        conn.close()

        total_inserted = 0
        for exec_row in executives:
            exec_id   = exec_row["id"]
            exec_name = exec_row["name"]

            stmts = get_statements_for_executive(exec_id)
            exec_inserted = 0

            for stmt in stmts:
                stmt_dict = dict(stmt)
                stmt_type = (stmt_dict.get("statement_type") or "").upper()

                if stmt_type not in _GUIDANCE_TYPES:
                    continue

                # Skip if we already extracted predictions for this statement
                if (exec_id, stmt_dict["id"]) in existing:
                    continue

                extracted = self._extract_from_statement(stmt_dict)
                for pred in extracted:
                    try:
                        insert_prediction(
                            executive_id   = exec_id,
                            statement_id   = stmt_dict["id"],
                            quarter        = stmt_dict["quarter"],
                            metric         = pred["metric"],
                            predicted_value= pred["predicted_value"],
                            direction      = pred["direction"],
                            outcome_quarter= "",
                        )
                        existing.add((exec_id, stmt_dict["id"]))
                        exec_inserted += 1
                    except Exception as e:
                        logger.warning(f"Failed to insert prediction: {e}")

            total_inserted += exec_inserted
            if exec_inserted:
                logger.debug(f"  {exec_name}: {exec_inserted} prediction(s) extracted.")

        logger.info(f"PredictionExtractor complete — {total_inserted} predictions inserted.")
        return total_inserted

    def _extract_from_statement(self, stmt: dict) -> list:
        """Run all regex patterns on one statement's text. Return list of prediction dicts."""
        text   = stmt.get("text", "")
        results = []
        seen_values = set()

        for pattern in _PREDICTION_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    value = float(match.group(1))
                except (IndexError, ValueError):
                    continue

                # Deduplicate same number in same statement
                if value in seen_values:
                    continue
                seen_values.add(value)

                # Extract a window of text around the match for context
                start  = max(0, match.start() - _WINDOW_CHARS // 2)
                end    = min(len(text), match.end() + _WINDOW_CHARS // 2)
                window = text[start:end]

                metric    = _map_metric(window)
                direction = _infer_direction(window, value)

                results.append({
                    "metric":          metric,
                    "predicted_value": value,
                    "direction":       direction,
                })

        return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. CredibilityScorer
# ─────────────────────────────────────────────────────────────────────────────

class CredibilityScorer:
    """
    Computes a composite credibility score [0–100] per executive based on:
      - Contradiction penalties  (HARD, SOFT, OMISSION from contradictions table)
      - Prediction accuracy      (verified predictions from predictions table)
    """

    def __init__(self, db_path: Path = DB_PATH):
        from storage.database import DB_PATH
        self.db_path = db_path

    def score_executive(self, executive_id: int) -> dict:
        """
        Compute the credibility score for one executive.
        Returns a structured result dict.
        """
        # ── 1. Fetch executive metadata ───────────────────────────────────
        conn = get_connection(self.db_path)
        exec_row = conn.execute(
            """
            SELECT e.id, e.name, e.role, c.name AS company_name
            FROM executives e
            JOIN companies c ON c.id = e.company_id
            WHERE e.id = ?
            """,
            (executive_id,),
        ).fetchone()
        conn.close()

        if not exec_row:
            logger.error(f"Executive ID {executive_id} not found.")
            return {}

        exec_name    = exec_row["name"]
        exec_role    = exec_row["role"]
        company_name = exec_row["company_name"]

        # ── 2. Contradiction counts ───────────────────────────────────────
        contradictions = get_contradictions_for_executive(executive_id, db_path=self.db_path)
        hard_count = soft_count = omit_count = 0
        for row in contradictions:
            ct = (row["contradiction_type"] or "").upper()
            r_dict = dict(row)
            status = (r_dict.get("review_status") or "APPROVED").upper()
            if ct == "HARD":
                # Only count APPROVED or LEGACY_APPROVED HARD contradictions
                if status in ("APPROVED", "LEGACY_APPROVED"):
                    hard_count += 1
            elif ct == "SOFT":
                soft_count += 1
            elif ct == "OMISSION":
                omit_count += 1

        # ── 3. Prediction accuracy (verified only) ────────────────────────
        predictions   = get_predictions_for_executive(executive_id, db_path=self.db_path)
        verified_preds = [p for p in predictions if p["verified"] == 1 and p["actual_value"] is not None]

        dir_correct = dir_wrong = 0
        magnitude_errors = []

        for pred in verified_preds:
            pred_val  = pred["predicted_value"]
            actual    = pred["actual_value"]
            direction = (pred["direction"] or "").lower()

            if pred_val is None or actual is None:
                continue

            # Direction accuracy
            delta = actual - pred_val
            if direction == "up" and delta > 0:
                dir_correct += 1
            elif direction == "down" and delta < 0:
                dir_correct += 1
            elif direction == "stable" and abs(delta) <= abs(pred_val) * 0.05:
                dir_correct += 1
            else:
                dir_wrong += 1

            # Magnitude error (% deviation from predicted)
            if pred_val != 0:
                pct_error = abs(delta) / abs(pred_val) * 100
                magnitude_errors.append(pct_error)

        total_verified = dir_correct + dir_wrong
        direction_accuracy  = (dir_correct / total_verified * 100) if total_verified > 0 else None
        avg_magnitude_error = (sum(magnitude_errors) / len(magnitude_errors)) if magnitude_errors else None

        # ── 4. Composite credibility score (penalty model) ─────────────────
        #
        # Base = 100
        # Subtract contradiction penalties
        # Add/subtract prediction outcomes
        # Clamp to [0, 100]
        score = 100
        score -= hard_count * PENALTY_HARD
        score -= soft_count * PENALTY_SOFT
        score -= omit_count * PENALTY_OMISSION
        score += dir_correct * REWARD_CORRECT
        score -= dir_wrong   * PENALTY_WRONG
        score  = max(0, min(100, score))

        # ── 5. Accuracy-based score (note.md formula) ─────────────────────
        # Only meaningful when verified predictions exist
        accuracy_score = None
        if direction_accuracy is not None and avg_magnitude_error is not None:
            magnitude_accuracy = max(0, 100 - avg_magnitude_error)
            accuracy_score = round(
                direction_accuracy * 0.70 + magnitude_accuracy * 0.30, 2
            )

        # ── 6. Risk tier ──────────────────────────────────────────────────
        if score >= 70:
            tier = "LOW RISK"
        elif score >= 50:
            tier = "MEDIUM RISK"
        else:
            tier = "HIGH RISK"

        total_statements   = get_statement_count_for_executive(executive_id, db_path=self.db_path)
        total_contradictions = hard_count + soft_count + omit_count

        return {
            "executive_id":          executive_id,
            "name":                  exec_name,
            "role":                  exec_role,
            "company":               company_name,
            # Contradiction signals
            "hard_contradictions":   hard_count,
            "soft_contradictions":   soft_count,
            "omission_contradictions": omit_count,
            "total_contradictions":  total_contradictions,
            # Prediction accuracy
            "total_predictions":     len(predictions),
            "verified_predictions":  total_verified,
            "direction_correct":     dir_correct,
            "direction_wrong":       dir_wrong,
            "direction_accuracy_pct": round(direction_accuracy, 2) if direction_accuracy is not None else None,
            "avg_magnitude_error_pct": round(avg_magnitude_error, 2) if avg_magnitude_error is not None else None,
            "accuracy_score":        accuracy_score,
            # Credibility
            "credibility_score":     score,
            "risk_tier":             tier,
            "total_statements":      total_statements,
        }

    def score_all(self) -> list:
        """Score all executives in the database. Returns list of result dicts."""
        executives = get_all_executives(db_path=self.db_path)
        results    = []
        for exec_row in executives:
            result = self.score_executive(exec_row["id"])
            if result:
                results.append(result)
        # Sort by credibility score ascending (most at-risk first)
        results.sort(key=lambda r: r["credibility_score"])
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Report formatter (used by run_credibility.py)
# ─────────────────────────────────────────────────────────────────────────────

def format_credibility_report(results: list) -> str:
    """
    Format a list of credibility score dicts into a human-readable report string.
    """
    if not results:
        return "No credibility data available. Run extraction and pipeline first."

    lines = []
    lines.append("=" * 80)
    lines.append("EXECUTIVE CREDIBILITY REPORT")
    lines.append("=" * 80)

    for r in results:
        lines.append(f"\n{'─' * 70}")
        lines.append(
            f"  {r['name']}  ({r['role']})  —  {r['company']}"
        )
        lines.append(f"  Credibility Score : {r['credibility_score']:>5.1f} / 100  [{r['risk_tier']}]")
        lines.append(f"  Total Statements  : {r['total_statements']}")

        # Contradictions
        lines.append(
            f"  Contradictions    : "
            f"HARD={r['hard_contradictions']}  "
            f"SOFT={r['soft_contradictions']}  "
            f"OMISSION={r['omission_contradictions']}"
        )

        # Prediction accuracy
        if r["verified_predictions"] > 0:
            lines.append(
                f"  Predictions       : "
                f"{r['verified_predictions']} verified  |  "
                f"Correct={r['direction_correct']}  Wrong={r['direction_wrong']}"
            )
            if r["direction_accuracy_pct"] is not None:
                lines.append(
                    f"  Direction Acc.    : {r['direction_accuracy_pct']:.1f}%  |  "
                    f"Avg Magnitude Err: "
                    + (f"{r['avg_magnitude_error_pct']:.1f}%" if r["avg_magnitude_error_pct"] is not None else "N/A")
                )
            if r["accuracy_score"] is not None:
                lines.append(f"  Accuracy Score    : {r['accuracy_score']:.1f} / 100")
        else:
            lines.append(
                f"  Predictions       : {r['total_predictions']} extracted  (0 verified — run --verify)"
            )

    lines.append(f"\n{'─' * 70}")
    lines.append("\nSUMMARY")
    lines.append(f"  Total executives scored : {len(results)}")
    low  = sum(1 for r in results if r['risk_tier'] == 'LOW RISK')
    med  = sum(1 for r in results if r['risk_tier'] == 'MEDIUM RISK')
    high = sum(1 for r in results if r['risk_tier'] == 'HIGH RISK')
    lines.append(f"  LOW RISK    : {low}")
    lines.append(f"  MEDIUM RISK : {med}")
    lines.append(f"  HIGH RISK   : {high}")
    lines.append("=" * 80)

    return "\n".join(lines)
