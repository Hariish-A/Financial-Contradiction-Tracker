"""
mcp_server/tools/contradictions.py
----------------------------------
MCP tools for querying detected contradictions (HARD, SOFT, OMISSION).
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.database import get_connection, DB_PATH

MAX_LIMIT = 100


def get_contradictions(
    company_name: Optional[str] = None,
    executive_name: Optional[str] = None,
    contradiction_type: Optional[str] = None,
    minimum_score: float = 0.0,
    review_status: Optional[str] = None,
    quarter: Optional[str] = None,
    limit: int = 50,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Get detected contradictions filtered by company, executive, type, score, review status, and quarter.
    """
    limit = max(1, min(limit, MAX_LIMIT))
    conn = get_connection(db_path)

    query = """
        SELECT
            c.id AS contradiction_id,
            c.contradiction_type,
            c.score,
            c.review_status,
            c.reviewer_name,
            c.review_notes,
            c.reviewed_at,
            c.decision_source,
            c.llm_verdict,
            c.llm_confidence,
            c.llm_explanation,
            c.graph_thread_id,
            c.details,
            sa.text AS statement_a_text,
            sa.quarter AS quarter_a,
            sa.year AS year_a,
            sb.text AS statement_b_text,
            sb.quarter AS quarter_b,
            sb.year AS year_b,
            e.name AS executive_name,
            e.role AS executive_role,
            co.name AS company_name,
            co.bse_code
        FROM contradictions c
        JOIN statements sa ON sa.id = c.statement_a_id
        JOIN statements sb ON sb.id = c.statement_b_id
        JOIN executives e ON e.id = sa.executive_id
        JOIN companies co ON co.id = sa.company_id
        WHERE c.score >= ?
    """
    params: List[Any] = [minimum_score]

    if company_name:
        query += " AND (co.name LIKE ? OR co.bse_code = ?)"
        params.extend([f"%{company_name}%", company_name])

    if executive_name:
        query += " AND e.name LIKE ?"
        params.append(f"%{executive_name}%")

    if contradiction_type:
        query += " AND UPPER(c.contradiction_type) = UPPER(?)"
        params.append(contradiction_type)

    if review_status:
        query += " AND UPPER(c.review_status) = UPPER(?)"
        params.append(review_status)

    if quarter:
        query += " AND (sa.quarter = ? OR sb.quarter = ?)"
        params.extend([quarter, quarter])

    query += " ORDER BY c.score DESC, c.id DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    output = []
    for r in rows:
        d = dict(r)
        try:
            d["details"] = json.loads(d["details"]) if d["details"] else {}
        except Exception:
            d["details"] = {}
        output.append(d)

    return output
