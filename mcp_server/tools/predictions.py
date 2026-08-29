"""
mcp_server/tools/predictions.py
-------------------------------
MCP tools for prediction status queries and actual value verification.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.database import get_connection, update_prediction_actual, DB_PATH
from credibility.scorer import CredibilityScorer

MAX_LIMIT = 100


def get_prediction_status(
    executive_id: Optional[int] = None,
    verified_only: Optional[bool] = None,
    limit: int = 50,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Get extracted guidance predictions and verification status.
    """
    limit = max(1, min(limit, MAX_LIMIT))
    conn = get_connection(db_path)

    query = """
        SELECT
            p.id AS prediction_id,
            p.executive_id,
            p.statement_id,
            p.quarter,
            p.metric,
            p.predicted_value,
            p.direction,
            p.actual_value,
            p.outcome_quarter,
            p.verified,
            e.name AS executive_name,
            e.role AS executive_role,
            co.name AS company_name
        FROM predictions p
        JOIN executives e ON e.id = p.executive_id
        JOIN companies co ON co.id = e.company_id
        WHERE 1=1
    """
    params: List[Any] = []

    if executive_id is not None:
        query += " AND p.executive_id = ?"
        params.append(executive_id)

    if verified_only is True:
        query += " AND p.verified = 1"
    elif verified_only is False:
        query += " AND (p.verified = 0 OR p.verified IS NULL)"

    query += " ORDER BY p.quarter DESC, p.id DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def verify_prediction_actual(
    prediction_id: int,
    actual_value: float,
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """
    Verify a guidance prediction with actual financial outcome and recompute executive credibility.
    """
    conn = get_connection(db_path)
    p_row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    conn.close()

    if not p_row:
        return {"success": False, "error": f"Prediction ID {prediction_id} not found."}

    initial_verified = bool(p_row["verified"])
    initial_actual = p_row["actual_value"]
    exec_id = p_row["executive_id"]

    update_prediction_actual(prediction_id, actual_value, verified=1, db_path=db_path)

    scorer = CredibilityScorer(db_path=db_path)
    updated_score = scorer.score_executive(exec_id)

    return {
        "success": True,
        "prediction_id": prediction_id,
        "executive_id": exec_id,
        "metric": p_row["metric"],
        "predicted_value": p_row["predicted_value"],
        "previous_actual_value": initial_actual,
        "new_actual_value": actual_value,
        "verified": True,
        "updated_credibility_score": updated_score.get("credibility_score"),
        "direction_accuracy_pct": updated_score.get("direction_accuracy_pct"),
    }
