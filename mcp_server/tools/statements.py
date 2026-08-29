"""
mcp_server/tools/statements.py
------------------------------
MCP tools for querying extracted executive guidance statements.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.database import get_connection, DB_PATH

MAX_LIMIT = 100


def query_statements(
    company_query: Optional[str] = None,
    executive_name: Optional[str] = None,
    quarter: Optional[str] = None,
    statement_type: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 50,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Query executive statements with structured filters.
    Limit is capped at 100 to prevent dumping large tables.
    """
    limit = max(1, min(limit, MAX_LIMIT))
    conn = get_connection(db_path)

    query = """
        SELECT
            s.id AS statement_id,
            s.quarter,
            s.year,
            s.text,
            s.statement_type,
            s.sentiment,
            s.sentiment_score,
            s.created_at,
            e.name AS executive_name,
            e.role AS executive_role,
            c.name AS company_name,
            c.bse_code
        FROM statements s
        JOIN executives e ON e.id = s.executive_id
        JOIN companies c ON c.id = s.company_id
        WHERE 1=1
    """
    params: List[Any] = []

    if company_query:
        query += " AND (c.name LIKE ? OR c.bse_code = ?)"
        params.extend([f"%{company_query}%", company_query])

    if executive_name:
        query += " AND e.name LIKE ?"
        params.append(f"%{executive_name}%")

    if quarter:
        query += " AND s.quarter = ?"
        params.append(quarter)

    if statement_type:
        query += " AND UPPER(s.statement_type) = UPPER(?)"
        params.append(statement_type)

    if sentiment:
        query += " AND LOWER(s.sentiment) = LOWER(?)"
        params.append(sentiment)

    query += " ORDER BY s.year DESC, s.quarter DESC, s.id DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(r) for r in rows]
