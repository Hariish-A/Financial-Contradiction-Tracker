"""
orchestration/service.py
------------------------
High-level service interface for running and resuming LangGraph HARD-contradiction workflows.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.types import Command
from orchestration.checkpointing import get_checkpointer, CHECKPOINT_DB_PATH
from orchestration.contradiction_graph import build_contradiction_graph
from storage.database import get_connection, DB_PATH

WORKFLOW_VERSION = "1.0"


def generate_thread_id(statement_a_id: int, statement_b_id: int, version: str = WORKFLOW_VERSION) -> str:
    """Generate a deterministic, idempotent graph thread identifier."""
    return f"hard_{statement_a_id}_{statement_b_id}_v{version.replace('.', '_')}"


def run_hard_contradiction_workflow(
    statement_a_id: int,
    statement_b_id: int,
    cosine_similarity: float = 0.0,
    db_path: Path = DB_PATH,
    checkpoint_db_path: Path = CHECKPOINT_DB_PATH,
) -> Dict[str, Any]:
    """
    Run or step through the LangGraph HARD contradiction workflow for a pair.
    Returns current state summary dict.
    """
    thread_id = generate_thread_id(statement_a_id, statement_b_id)
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "statement_a_id": statement_a_id,
        "statement_b_id": statement_b_id,
        "statement_a_text": "",
        "statement_b_text": "",
        "company_name": "",
        "executive_name": "",
        "executive_role": "",
        "quarter_a": "",
        "quarter_b": "",
        "year_a": 0,
        "year_b": 0,
        "cosine_similarity": cosine_similarity,
        "nli_contradiction_score": 0.0,
        "nli_neutral_score": 0.0,
        "nli_entailment_score": 0.0,
        "nli_verdict": "",
        "routing_decision": "",
        "llm_verdict": None,
        "llm_confidence": None,
        "llm_explanation": None,
        "llm_metadata": None,
        "review_status": "NOT_REQUIRED",
        "decision_source": "NLI",
        "reviewer_name": None,
        "review_notes": None,
        "reviewed_at": None,
        "db_path": str(db_path),
        "graph_thread_id": thread_id,
        "workflow_version": WORKFLOW_VERSION,
        "error_message": None,
        "is_completed": False,
    }

    with get_checkpointer(checkpoint_db_path) as checkpointer:
        app = build_contradiction_graph(checkpointer=checkpointer)

        # Execute graph until completion or interrupt
        result_state = app.invoke(initial_state, config=config)

    return result_state


def resume_human_review_workflow(
    thread_id: str,
    approved: bool,
    reviewer_name: str = "human",
    review_notes: str = "",
    checkpoint_db_path: Path = CHECKPOINT_DB_PATH,
) -> Dict[str, Any]:
    """
    Resume an interrupted LangGraph workflow thread with a human review decision.
    """
    config = {"configurable": {"thread_id": thread_id}}

    import datetime
    reviewed_at = datetime.datetime.now().isoformat()

    resume_payload = {
        "approved": approved,
        "reviewer_name": reviewer_name,
        "review_notes": review_notes,
        "reviewed_at": reviewed_at,
    }

    with get_checkpointer(checkpoint_db_path) as checkpointer:
        app = build_contradiction_graph(checkpointer=checkpointer)
        result_state = app.invoke(Command(resume=resume_payload), config=config)

    return result_state


def list_pending_reviews(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """
    Query database for pending review candidates awaiting human decision.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT
            c.id AS contradiction_id,
            c.statement_a_id,
            c.statement_b_id,
            c.score AS nli_score,
            c.review_status,
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
            co.name AS company_name
        FROM contradictions c
        JOIN statements sa ON sa.id = c.statement_a_id
        JOIN statements sb ON sb.id = c.statement_b_id
        JOIN executives e ON e.id = sa.executive_id
        JOIN companies co ON co.id = sa.company_id
        WHERE c.review_status = 'PENDING'
        ORDER BY c.score DESC
        """
    ).fetchall()
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
