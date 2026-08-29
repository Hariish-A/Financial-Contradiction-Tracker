"""
orchestration/contradiction_graph.py
------------------------------------
LangGraph state graph definition for stateful HARD contradiction adjudication.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestration.state import ContradictionState
from orchestration.routing import route_nli_result, route_llm_result
from orchestration.llm_judge import evaluate_contradiction_pair
from contradiction.nli_scorer import score_contradiction
from contradiction.services import persist_contradiction_record
from storage.database import get_connection, DB_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────────────────────────────────────

def load_pair_node(state: ContradictionState) -> Dict[str, Any]:
    """Node 1: Load statement A and B texts and metadata from SQLite if missing."""
    db_path = Path(state["db_path"]) if state.get("db_path") else DB_PATH
    conn = get_connection(db_path)
    
    sa_row = conn.execute(
        """
        SELECT s.*, e.name AS executive_name, e.role AS executive_role, c.name AS company_name
        FROM statements s
        JOIN executives e ON e.id = s.executive_id
        JOIN companies c ON c.id = s.company_id
        WHERE s.id = ?
        """,
        (state["statement_a_id"],),
    ).fetchone()

    sb_row = conn.execute(
        "SELECT * FROM statements WHERE id = ?",
        (state["statement_b_id"],),
    ).fetchone()
    conn.close()

    updates = {}
    if sa_row:
        updates["statement_a_text"] = sa_row["text"]
        updates["company_name"] = sa_row["company_name"]
        updates["executive_name"] = sa_row["executive_name"]
        updates["executive_role"] = sa_row["executive_role"]
        updates["quarter_a"] = sa_row["quarter"]
        updates["year_a"] = sa_row["year"]

    if sb_row:
        updates["statement_b_text"] = sb_row["text"]
        updates["quarter_b"] = sb_row["quarter"]
        updates["year_b"] = sb_row["year"]

    return updates


def score_nli_node(state: ContradictionState) -> Dict[str, Any]:
    """Node 2: Calculate NLI score distribution using DeBERTa cross-encoder."""
    nli_res = score_contradiction(state["statement_a_text"], state["statement_b_text"])
    return {
        "nli_contradiction_score": round(nli_res["contradiction_score"], 4),
        "nli_neutral_score": round(nli_res["neutral_score"], 4),
        "nli_entailment_score": round(nli_res["entailment_score"], 4),
        "nli_verdict": nli_res["verdict"],
    }


def route_nli_node(state: ContradictionState) -> Dict[str, Any]:
    """Node 3: Determine routing path based on NLI score distribution."""
    decision = route_nli_result(
        nli_contradiction_prob=state["nli_contradiction_score"],
        nli_verdict=state["nli_verdict"],
    )
    updates = {"routing_decision": decision}
    if decision == "AUTO_HARD":
        updates["decision_source"] = "NLI"
    return updates


def llm_judge_node(state: ContradictionState) -> Dict[str, Any]:
    """Node 4: Conditional escalation to LLM judge for ambiguous NLI results."""
    logger.info(f"Escalating pair ({state['statement_a_id']}, {state['statement_b_id']}) to LLM Judge...")
    llm_res = evaluate_contradiction_pair(
        statement_a=state["statement_a_text"],
        statement_b=state["statement_b_text"],
        company=state.get("company_name", ""),
        executive=state.get("executive_name", ""),
        quarter_a=state.get("quarter_a", ""),
        quarter_b=state.get("quarter_b", ""),
    )

    next_decision = route_llm_result(llm_res.verdict, llm_res.confidence)
    return {
        "llm_verdict": llm_res.verdict,
        "llm_confidence": llm_res.confidence,
        "llm_explanation": llm_res.explanation,
        "llm_metadata": {"supporting_excerpts": llm_res.supporting_excerpts},
        "routing_decision": next_decision,
        "decision_source": "LLM",
    }


def prepare_human_review_node(state: ContradictionState) -> Dict[str, Any]:
    """Node 5: Prepare pending status and payload before human review interrupt."""
    return {"review_status": "PENDING"}


def human_review_interrupt_node(state: ContradictionState) -> Dict[str, Any]:
    """
    Node 6: Human-in-the-loop interrupt.
    Pauses execution until human reviewer approves or rejects.
    """
    payload = {
        "statement_a_id": state["statement_a_id"],
        "statement_b_id": state["statement_b_id"],
        "statement_a_text": state["statement_a_text"],
        "statement_b_text": state["statement_b_text"],
        "company_name": state.get("company_name", ""),
        "executive_name": state.get("executive_name", ""),
        "quarter_a": state.get("quarter_a", ""),
        "quarter_b": state.get("quarter_b", ""),
        "cosine_similarity": state.get("cosine_similarity", 0.0),
        "nli_scores": {
            "contradiction": state.get("nli_contradiction_score", 0.0),
            "neutral": state.get("nli_neutral_score", 0.0),
            "entailment": state.get("nli_entailment_score", 0.0),
        },
        "llm_evidence": {
            "verdict": state.get("llm_verdict"),
            "confidence": state.get("llm_confidence"),
            "explanation": state.get("llm_explanation"),
        },
        "expected_credibility_penalty": 20,
    }

    # LangGraph interrupt pauses workflow and returns payload to caller
    human_response = interrupt(payload)

    # When resumed, human_response contains review decision
    approved = human_response.get("approved", False) if isinstance(human_response, dict) else False
    reviewer_name = human_response.get("reviewer_name", "human") if isinstance(human_response, dict) else "human"
    review_notes = human_response.get("review_notes", "") if isinstance(human_response, dict) else ""
    reviewed_at = human_response.get("reviewed_at", "") if isinstance(human_response, dict) else ""

    if approved:
        return {
            "review_status": "APPROVED",
            "reviewer_name": reviewer_name,
            "review_notes": review_notes,
            "reviewed_at": reviewed_at,
        }
    else:
        return {
            "review_status": "REJECTED",
            "reviewer_name": reviewer_name,
            "review_notes": review_notes,
            "reviewed_at": reviewed_at,
        }


def persist_approved_contradiction_node(state: ContradictionState) -> Dict[str, Any]:
    """Node 7: Persist APPROVED HARD contradiction in database."""
    db_path = Path(state["db_path"]) if state.get("db_path") else DB_PATH
    details = {
        "nli_contradiction_score": state["nli_contradiction_score"],
        "nli_neutral_score": state["nli_neutral_score"],
        "nli_entailment_score": state["nli_entailment_score"],
        "cosine_similarity": state.get("cosine_similarity", 0.0),
        "quarter_a": state.get("quarter_a", ""),
        "quarter_b": state.get("quarter_b", ""),
    }

    persist_contradiction_record(
        statement_a_id=state["statement_a_id"],
        statement_b_id=state["statement_b_id"],
        contradiction_type="HARD",
        score=state["nli_contradiction_score"],
        details=details,
        review_status="APPROVED",
        reviewer_name=state.get("reviewer_name"),
        review_notes=state.get("review_notes"),
        reviewed_at=state.get("reviewed_at"),
        decision_source=state.get("decision_source", "HUMAN"),
        nli_scores_json={"c": state["nli_contradiction_score"], "n": state["nli_neutral_score"], "e": state["nli_entailment_score"]},
        llm_verdict=state.get("llm_verdict"),
        llm_confidence=state.get("llm_confidence"),
        llm_explanation=state.get("llm_explanation"),
        llm_metadata_json=state.get("llm_metadata"),
        graph_thread_id=state.get("graph_thread_id"),
        workflow_version=state.get("workflow_version", "1.0"),
        reviewed=1,
        db_path=db_path,
    )
    return {"is_completed": True}


def persist_rejected_audit_node(state: ContradictionState) -> Dict[str, Any]:
    """Node 8: Persist REJECTED or AUTO_REJECT audit record in database if needed."""
    db_path = Path(state["db_path"]) if state.get("db_path") else DB_PATH
    status = state.get("review_status", "NOT_REQUIRED")
    if status not in ("REJECTED", "NOT_REQUIRED"):
        status = "REJECTED"

    details = {
        "nli_contradiction_score": state.get("nli_contradiction_score", 0.0),
        "cosine_similarity": state.get("cosine_similarity", 0.0),
        "reason": "Dismissed by routing or rejected by human review.",
    }

    try:
        persist_contradiction_record(
            statement_a_id=state["statement_a_id"],
            statement_b_id=state["statement_b_id"],
            contradiction_type="HARD",
            score=state.get("nli_contradiction_score", 0.0),
            details=details,
            review_status=status,
            reviewer_name=state.get("reviewer_name"),
            review_notes=state.get("review_notes"),
            reviewed_at=state.get("reviewed_at"),
            decision_source=state.get("decision_source", "NLI"),
            nli_scores_json={"c": state.get("nli_contradiction_score", 0.0)},
            llm_verdict=state.get("llm_verdict"),
            llm_confidence=state.get("llm_confidence"),
            llm_explanation=state.get("llm_explanation"),
            graph_thread_id=state.get("graph_thread_id"),
            workflow_version=state.get("workflow_version", "1.0"),
            reviewed=0,
            db_path=db_path,
        )
    except Exception as e:
        logger.debug(f"Audit record persistence skipped: {e}")

    return {"is_completed": True}


# ─────────────────────────────────────────────────────────────────────────────
# Edge Routing Functions
# ─────────────────────────────────────────────────────────────────────────────

def _nli_branch(state: ContradictionState) -> str:
    decision = state.get("routing_decision", "")
    if decision == "AUTO_HARD":
        return "prepare_human_review"
    elif decision == "LLM_EVALUATE":
        return "llm_judge"
    else:
        return "persist_rejected_audit"


def _llm_branch(state: ContradictionState) -> str:
    decision = state.get("routing_decision", "")
    if decision == "HUMAN_REVIEW":
        return "prepare_human_review"
    else:
        return "persist_rejected_audit"


def _review_branch(state: ContradictionState) -> str:
    status = state.get("review_status", "")
    if status == "APPROVED":
        return "persist_approved"
    else:
        return "persist_rejected_audit"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_contradiction_graph(checkpointer=None):
    """
    Construct and compile the stateful LangGraph HARD contradiction workflow graph.
    """
    workflow = StateGraph(ContradictionState)

    # Add Nodes
    workflow.add_node("load_pair", load_pair_node)
    workflow.add_node("score_nli", score_nli_node)
    workflow.add_node("route_nli", route_nli_node)
    workflow.add_node("llm_judge", llm_judge_node)
    workflow.add_node("prepare_human_review", prepare_human_review_node)
    workflow.add_node("human_review_interrupt", human_review_interrupt_node)
    workflow.add_node("persist_approved", persist_approved_contradiction_node)
    workflow.add_node("persist_rejected_audit", persist_rejected_audit_node)

    # Add Edges
    workflow.add_edge(START, "load_pair")
    workflow.add_edge("load_pair", "score_nli")
    workflow.add_edge("score_nli", "route_nli")

    # Conditional Branch from route_nli
    workflow.add_conditional_edges(
        "route_nli",
        _nli_branch,
        {
            "prepare_human_review": "prepare_human_review",
            "llm_judge": "llm_judge",
            "persist_rejected_audit": "persist_rejected_audit",
        },
    )

    # Conditional Branch from llm_judge
    workflow.add_conditional_edges(
        "llm_judge",
        _llm_branch,
        {
            "prepare_human_review": "prepare_human_review",
            "persist_rejected_audit": "persist_rejected_audit",
        },
    )

    workflow.add_edge("prepare_human_review", "human_review_interrupt")

    # Conditional Branch from human_review_interrupt
    workflow.add_conditional_edges(
        "human_review_interrupt",
        _review_branch,
        {
            "persist_approved": "persist_approved",
            "persist_rejected_audit": "persist_rejected_audit",
        },
    )

    workflow.add_edge("persist_approved", END)
    workflow.add_edge("persist_rejected_audit", END)

    return workflow.compile(checkpointer=checkpointer)
