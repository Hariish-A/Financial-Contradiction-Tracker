"""
mcp_server/tools/search.py
--------------------------
MCP tool for semantic statement search via StatementIndex and FAISS retrieval.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contradiction.embeddings import StatementIndex
from storage.database import get_connection, DB_PATH

MAX_LIMIT = 50


def find_similar_statements(
    executive_id: int,
    query_text: str,
    top_k: int = 5,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Find semantically similar statements for an executive using FAISS embeddings.
    """
    top_k = max(1, min(top_k, MAX_LIMIT))
    index = StatementIndex(executive_id)
    results = index.retrieve_similar(query_text, top_k=top_k)

    output = []
    for stmt, score in results:
        output.append({
            "statement_id": stmt.get("id"),
            "executive_id": stmt.get("executive_id"),
            "company_id": stmt.get("company_id"),
            "quarter": stmt.get("quarter"),
            "year": stmt.get("year"),
            "statement_type": stmt.get("statement_type"),
            "sentiment": stmt.get("sentiment"),
            "text": stmt.get("text"),
            "cosine_similarity": round(float(score), 4),
        })

    return output
