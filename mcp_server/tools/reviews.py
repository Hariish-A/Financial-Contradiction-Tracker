"""
mcp_server/tools/reviews.py
---------------------------
MCP tools for listing, approving, and rejecting HARD contradiction candidates.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.service import list_pending_reviews as list_pending_svc, resume_human_review_workflow
from credibility.scorer import CredibilityScorer
from storage.database import get_connection, DB_PATH


def list_pending_reviews(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """
    List pending review items with evidence (NLI scores, LLM verdict, statement texts).
    """
    return list_pending_svc(db_path=db_path)


def approve_contradiction(
    contradiction_id: int,
    reviewer_name: str = "mcp_agent",
    review_notes: str = "Approved via MCP tool",
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """
    Approve a pending HARD contradiction candidate.
    Validates identifier, updates graph workflow/database status, and returns before/after status.
    """
    conn = get_connection(db_path)
    c_row = conn.execute("SELECT * FROM contradictions WHERE id = ?", (contradiction_id,)).fetchone()
    conn.close()

    if not c_row:
        return {"success": False, "error": f"Contradiction ID {contradiction_id} not found."}

    initial_status = c_row["review_status"]
    thread_id = c_row["graph_thread_id"]

    if thread_id and thread_id.startswith("hard_"):
        try:
            resume_human_review_workflow(
                thread_id=thread_id,
                approved=True,
                reviewer_name=reviewer_name,
                review_notes=review_notes,
            )
        except Exception:
            _direct_db_update(contradiction_id, "APPROVED", reviewer_name, review_notes, 1, db_path)
    else:
        _direct_db_update(contradiction_id, "APPROVED", reviewer_name, review_notes, 1, db_path)

    # Re-fetch after state
    conn = get_connection(db_path)
    after_row = conn.execute(
        """
        SELECT c.*, sa.executive_id
        FROM contradictions c
        JOIN statements sa ON sa.id = c.statement_a_id
        WHERE c.id = ?
        """,
        (contradiction_id,),
    ).fetchone()
    conn.close()

    exec_id = after_row["executive_id"]
    scorer = CredibilityScorer(db_path=db_path)
    updated_score = scorer.score_executive(exec_id)

    return {
        "success": True,
        "contradiction_id": contradiction_id,
        "previous_status": initial_status,
        "current_status": after_row["review_status"],
        "reviewer_name": reviewer_name,
        "review_notes": review_notes,
        "executive_id": exec_id,
        "updated_credibility_score": updated_score.get("credibility_score"),
    }


def reject_contradiction(
    contradiction_id: int,
    reviewer_name: str = "mcp_agent",
    review_notes: str = "Rejected via MCP tool",
    db_path: Path = DB_PATH,
) -> Dict[str, Any]:
    """
    Reject a pending HARD contradiction candidate.
    Validates identifier, updates graph workflow/database status, and returns before/after status.
    """
    conn = get_connection(db_path)
    c_row = conn.execute("SELECT * FROM contradictions WHERE id = ?", (contradiction_id,)).fetchone()
    conn.close()

    if not c_row:
        return {"success": False, "error": f"Contradiction ID {contradiction_id} not found."}

    initial_status = c_row["review_status"]
    thread_id = c_row["graph_thread_id"]

    if thread_id and thread_id.startswith("hard_"):
        try:
            resume_human_review_workflow(
                thread_id=thread_id,
                approved=False,
                reviewer_name=reviewer_name,
                review_notes=review_notes,
            )
        except Exception:
            _direct_db_update(contradiction_id, "REJECTED", reviewer_name, review_notes, 0, db_path)
    else:
        _direct_db_update(contradiction_id, "REJECTED", reviewer_name, review_notes, 0, db_path)

    conn = get_connection(db_path)
    after_row = conn.execute(
        """
        SELECT c.*, sa.executive_id
        FROM contradictions c
        JOIN statements sa ON sa.id = c.statement_a_id
        WHERE c.id = ?
        """,
        (contradiction_id,),
    ).fetchone()
    conn.close()

    exec_id = after_row["executive_id"]
    scorer = CredibilityScorer(db_path=db_path)
    updated_score = scorer.score_executive(exec_id)

    return {
        "success": True,
        "contradiction_id": contradiction_id,
        "previous_status": initial_status,
        "current_status": after_row["review_status"],
        "reviewer_name": reviewer_name,
        "review_notes": review_notes,
        "executive_id": exec_id,
        "updated_credibility_score": updated_score.get("credibility_score"),
    }


def _direct_db_update(contradiction_id: int, status: str, reviewer: str, notes: str, reviewed: int, db_path: Path):
    import datetime
    conn = get_connection(db_path)
    conn.execute(
        """
        UPDATE contradictions
        SET review_status = ?,
            reviewer_name = ?,
            review_notes = ?,
            reviewed_at = ?,
            reviewed = ?
        WHERE id = ?
        """,
        (status, reviewer, notes, datetime.datetime.now().isoformat(), reviewed, contradiction_id),
    )
    conn.commit()
    conn.close()
