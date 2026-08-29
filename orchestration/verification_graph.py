"""
orchestration/verification_graph.py
-----------------------------------
LangGraph state graph for prediction-versus-actual financial verification.
Links qualitative/quantitative guidance predictions with actual quarterly financial outcomes.
"""

import sys
from pathlib import Path
from typing import TypedDict, Optional, Dict, Any
from loguru import logger

from langgraph.graph import StateGraph, START, END

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.database import get_connection, update_prediction_actual, DB_PATH
from storage.financial_actuals import get_financial_actuals, fetch_and_store_screener_actuals
from credibility.scorer import CredibilityScorer


class VerificationState(TypedDict):
    prediction_id: int
    executive_id: int
    company_id: int
    ticker: str
    prediction_quarter: str
    outcome_quarter: str
    metric: str
    predicted_value: Optional[float]
    direction: str
    actual_value: Optional[float]
    matched_confidence: float
    is_direction_correct: Optional[bool]
    magnitude_error_pct: Optional[float]
    needs_human_review: bool
    is_verified: bool
    db_path: Optional[str]


def load_prediction_node(state: VerificationState) -> Dict[str, Any]:
    """Node 1: Load pending prediction details."""
    db_path = Path(state["db_path"]) if state.get("db_path") else DB_PATH
    conn = get_connection(db_path)
    
    p_row = conn.execute(
        """
        SELECT p.*, e.company_id, c.bse_code, c.name AS company_name
        FROM predictions p
        JOIN executives e ON e.id = p.executive_id
        JOIN companies c ON c.id = e.company_id
        WHERE p.id = ?
        """,
        (state["prediction_id"],),
    ).fetchone()
    conn.close()

    if not p_row:
        return {"needs_human_review": True}

    outcome_q = p_row["outcome_quarter"] or p_row["quarter"]
    ticker = p_row["bse_code"]

    return {
        "executive_id": p_row["executive_id"],
        "company_id": p_row["company_id"],
        "ticker": ticker,
        "prediction_quarter": p_row["quarter"],
        "outcome_quarter": outcome_q,
        "metric": p_row["metric"],
        "predicted_value": p_row["predicted_value"],
        "direction": p_row["direction"] or "up",
    }


def fetch_actuals_node(state: VerificationState) -> Dict[str, Any]:
    """Node 2 & 3: Fetch stored actuals or attempt Screener fetch."""
    db_path = Path(state["db_path"]) if state.get("db_path") else DB_PATH
    comp_id = state.get("company_id", 0)
    outcome_q = state.get("outcome_quarter", "")
    metric = state.get("metric", "")

    actuals = get_financial_actuals(comp_id, quarter=outcome_q, metric=metric, db_path=db_path)

    if not actuals and state.get("ticker"):
        try:
            fetch_and_store_screener_actuals(state["ticker"], comp_id, db_path=db_path)
            actuals = get_financial_actuals(comp_id, quarter=outcome_q, metric=metric, db_path=db_path)
        except Exception as exc:
            logger.warning(f"Screener fetch failed during verification: {exc}")

    if actuals:
        val = actuals[0]["value"]
        return {"actual_value": val, "matched_confidence": 0.95}

    # No actual financial value found -> escalate to human review (never invent numbers)
    return {"actual_value": None, "matched_confidence": 0.0, "needs_human_review": True}


def compare_outcomes_node(state: VerificationState) -> Dict[str, Any]:
    """Node 4, 5, 6: Compare prediction direction and magnitude against actual outcome."""
    pred_val = state.get("predicted_value")
    actual_val = state.get("actual_value")
    direction = (state.get("direction") or "up").lower()

    if actual_val is None:
        return {"needs_human_review": True}

    delta = actual_val - (pred_val if pred_val is not None else 0.0)

    is_correct = False
    if direction == "up" and delta > 0:
        is_correct = True
    elif direction == "down" and delta < 0:
        is_correct = True
    elif direction == "stable" and (pred_val is None or abs(delta) <= abs(pred_val) * 0.05):
        is_correct = True

    mag_err = None
    if pred_val and pred_val != 0:
        mag_err = round(abs(delta) / abs(pred_val) * 100, 2)

    return {
        "is_direction_correct": is_correct,
        "magnitude_error_pct": mag_err,
    }


def persist_verification_node(state: VerificationState) -> Dict[str, Any]:
    """Node 8 & 9: Persist verification outcome and recompute executive credibility."""
    db_path = Path(state["db_path"]) if state.get("db_path") else DB_PATH
    pred_id = state["prediction_id"]
    actual_val = state.get("actual_value")

    if actual_val is not None:
        update_prediction_actual(pred_id, actual_val, verified=1, db_path=db_path)
        scorer = CredibilityScorer(db_path=db_path)
        scorer.score_executive(state["executive_id"])
        return {"is_verified": True}

    return {"is_verified": False}


def build_verification_graph():
    """Build and compile the prediction verification LangGraph workflow."""
    workflow = StateGraph(VerificationState)

    workflow.add_node("load_prediction", load_prediction_node)
    workflow.add_node("fetch_actuals", fetch_actuals_node)
    workflow.add_node("compare_outcomes", compare_outcomes_node)
    workflow.add_node("persist_verification", persist_verification_node)

    workflow.add_edge(START, "load_prediction")
    workflow.add_edge("load_prediction", "fetch_actuals")
    workflow.add_edge("fetch_actuals", "compare_outcomes")
    workflow.add_edge("compare_outcomes", "persist_verification")
    workflow.add_edge("persist_verification", END)

    return workflow.compile()


def run_prediction_verification(
    prediction_id: int,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """
    Run prediction verification workflow for a prediction ID.
    """
    app = build_verification_graph()
    initial_state = {
        "prediction_id": prediction_id,
        "executive_id": 0,
        "company_id": 0,
        "ticker": "",
        "prediction_quarter": "",
        "outcome_quarter": "",
        "metric": "",
        "predicted_value": None,
        "direction": "up",
        "actual_value": None,
        "matched_confidence": 0.0,
        "is_direction_correct": None,
        "magnitude_error_pct": None,
        "needs_human_review": False,
        "is_verified": False,
        "db_path": str(db_path),
    }

    return app.invoke(initial_state)
