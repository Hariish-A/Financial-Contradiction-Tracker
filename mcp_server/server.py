"""
mcp_server/server.py
--------------------
Standards-compliant Model Context Protocol (MCP) server for ContraGuard.
Built with Python MCP SDK (`mcp.server.mcpserver.MCPServer`).

Exposes domain services as 9 typed, bounded tools over stdio transport.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer

from mcp_server.tools.statements import query_statements as query_stmts_impl
from mcp_server.tools.contradictions import get_contradictions as get_contra_impl
from mcp_server.tools.credibility import get_credibility_score as get_cred_impl
from mcp_server.tools.search import find_similar_statements as search_impl
from mcp_server.tools.reviews import (
    list_pending_reviews as list_pending_impl,
    approve_contradiction as approve_impl,
    reject_contradiction as reject_impl,
)
from mcp_server.tools.predictions import (
    get_prediction_status as get_pred_impl,
    verify_prediction_actual as verify_pred_impl,
)

mcp = MCPServer("ContraGuard-MCP-Server")


# ─────────────────────────────────────────────────────────────────────────────
# 1. query_statements
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="query_statements",
    description="Query extracted executive guidance statements with optional filters for company name/BSE code, executive name, quarter, statement_type, or sentiment.",
)
def query_statements(
    company_query: Optional[str] = None,
    executive_name: Optional[str] = None,
    quarter: Optional[str] = None,
    statement_type: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    return query_stmts_impl(
        company_query=company_query,
        executive_name=executive_name,
        quarter=quarter,
        statement_type=statement_type,
        sentiment=sentiment,
        limit=limit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. get_contradictions
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="get_contradictions",
    description="Get detected contradictions filtered by company, executive, contradiction type (HARD, SOFT, OMISSION), minimum score, review status, and quarter.",
)
def get_contradictions(
    company_name: Optional[str] = None,
    executive_name: Optional[str] = None,
    contradiction_type: Optional[str] = None,
    minimum_score: float = 0.0,
    review_status: Optional[str] = None,
    quarter: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    return get_contra_impl(
        company_name=company_name,
        executive_name=executive_name,
        contradiction_type=contradiction_type,
        minimum_score=minimum_score,
        review_status=review_status,
        quarter=quarter,
        limit=limit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. get_credibility_score
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="get_credibility_score",
    description="Get calculated credibility score(s) and risk breakdown for a specific executive or all executives.",
)
def get_credibility_score(
    executive_id: Optional[int] = None,
    executive_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return get_cred_impl(
        executive_id=executive_id,
        executive_name=executive_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. find_similar_statements
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="find_similar_statements",
    description="Find semantically similar statements for an executive using FAISS embedding vector retrieval.",
)
def find_similar_statements(
    executive_id: int,
    query_text: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    return search_impl(
        executive_id=executive_id,
        query_text=query_text,
        top_k=top_k,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. list_pending_reviews
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="list_pending_reviews",
    description="Return pending HARD contradiction review items awaiting human decision along with evidence (NLI scores, LLM verdict).",
)
def list_pending_reviews() -> List[Dict[str, Any]]:
    return list_pending_impl()


# ─────────────────────────────────────────────────────────────────────────────
# 6. get_prediction_status
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="get_prediction_status",
    description="Return extracted quantitative guidance predictions and their verification status.",
)
def get_prediction_status(
    executive_id: Optional[int] = None,
    verified_only: Optional[bool] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    return get_pred_impl(
        executive_id=executive_id,
        verified_only=verified_only,
        limit=limit,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. approve_contradiction
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="approve_contradiction",
    description="Approve a candidate HARD contradiction in the human review queue. Deducts 20 points from executive credibility.",
)
def approve_contradiction(
    contradiction_id: int,
    reviewer_name: str = "mcp_agent",
    review_notes: str = "Approved via MCP tool",
) -> Dict[str, Any]:
    return approve_impl(
        contradiction_id=contradiction_id,
        reviewer_name=reviewer_name,
        review_notes=review_notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. reject_contradiction
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="reject_contradiction",
    description="Reject a candidate HARD contradiction candidate in the human review queue. Leaves credibility unchanged.",
)
def reject_contradiction(
    contradiction_id: int,
    reviewer_name: str = "mcp_agent",
    review_notes: str = "Rejected via MCP tool",
) -> Dict[str, Any]:
    return reject_impl(
        contradiction_id=contradiction_id,
        reviewer_name=reviewer_name,
        review_notes=review_notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. verify_prediction_actual
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="verify_prediction_actual",
    description="Record actual reported financial metric outcome against a prediction and recompute executive credibility score.",
)
def verify_prediction_actual(
    prediction_id: int,
    actual_value: float,
) -> Dict[str, Any]:
    return verify_pred_impl(
        prediction_id=prediction_id,
        actual_value=actual_value,
    )


def main():
    """Run the ContraGuard stdio MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
