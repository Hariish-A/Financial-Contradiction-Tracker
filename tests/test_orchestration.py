"""
tests/test_orchestration.py
----------------------------
Unit and integration tests for LangGraph stateful orchestration, routing thresholds,
LLM judge fallback, interrupt/resume mechanics, and SQLite checkpointer persistence.
"""

import pytest
from pathlib import Path

from storage.database import init_db, get_connection, upsert_company, upsert_executive, insert_statement
from orchestration.routing import route_nli_result, route_llm_result, NLI_AUTO_REJECT_MAX, NLI_AUTO_HARD_MIN
from orchestration.llm_judge import evaluate_contradiction_pair, LLMJudgeResult
from orchestration.service import run_hard_contradiction_workflow, resume_human_review_workflow, generate_thread_id
from credibility.scorer import CredibilityScorer


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "graph_test_tracker.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def checkpoint_db(tmp_path: Path) -> Path:
    return tmp_path / "test_checkpoints.db"


def test_routing_threshold_boundaries():
    """Verify routing policy at and around all threshold boundaries."""
    # <= 0.20 auto reject
    assert route_nli_result(0.15, "neutral") == "AUTO_REJECT"
    assert route_nli_result(0.20, "neutral") == "AUTO_REJECT"

    # >= 0.80 auto hard candidate
    assert route_nli_result(0.80, "contradiction") == "AUTO_HARD"
    assert route_nli_result(0.95, "contradiction") == "AUTO_HARD"

    # Ambiguous scores (0.20, 0.80) -> LLM evaluate
    assert route_nli_result(0.50, "neutral") == "LLM_EVALUATE"
    assert route_nli_result(0.65, "contradiction") == "LLM_EVALUATE"

    # LLM verdict routing
    assert route_llm_result("contradiction", 0.90) == "HUMAN_REVIEW"
    assert route_llm_result("consistent", 0.85) == "AUTO_REJECT"
    assert route_llm_result("uncertain", 0.40) == "HUMAN_REVIEW"


def test_llm_judge_fake_and_fallback(monkeypatch):
    """Verify LLM judge fake behavior and fallback on missing provider or error."""
    # Test fake judge
    res = evaluate_contradiction_pair(
        "We expect 18% revenue growth in the next quarter.",
        "We are revising our guidance to 8% for the quarter."
    )
    assert res.verdict == "contradiction"
    assert res.confidence >= 0.80

    # Test error fallback
    monkeypatch.setenv("LLM_PROVIDER", "invalid_provider")
    res_err = evaluate_contradiction_pair("Text A", "Text B")
    assert res_err.verdict == "uncertain"
    assert res_err.confidence == 0.0


def test_langgraph_interrupt_and_resume(temp_db: Path, checkpoint_db: Path):
    """
    Integration test: candidate pair -> graph -> interrupt at human review -> resume approval -> approved record.
    """
    # 1. Setup DB fixtures
    comp_id = upsert_company("Infosys", "500209", "IT", db_path=temp_db)
    exec_id = upsert_executive("Nilanjan Roy", "CFO", comp_id, db_path=temp_db)

    stmt_a_id = insert_statement(
        executive_id=exec_id,
        company_id=comp_id,
        transcript_id=None,
        quarter="Q1FY24",
        year=2024,
        text="We expect 18% revenue growth in the next quarter.",
        db_path=temp_db,
    )
    stmt_b_id = insert_statement(
        executive_id=exec_id,
        company_id=comp_id,
        transcript_id=None,
        quarter="Q2FY24",
        year=2024,
        text="We are revising our guidance to 8% for the quarter.",
        db_path=temp_db,
    )

    # 2. Run workflow -> should reach human_review_interrupt and pause
    state = run_hard_contradiction_workflow(
        statement_a_id=stmt_a_id,
        statement_b_id=stmt_b_id,
        cosine_similarity=0.85,
        db_path=temp_db,
        checkpoint_db_path=checkpoint_db,
    )

    thread_id = generate_thread_id(stmt_a_id, stmt_b_id)
    assert state.get("review_status") == "PENDING"

    # Verify credibility remains 100 before approval
    scorer = CredibilityScorer(db_path=temp_db)
    assert scorer.score_executive(exec_id)["credibility_score"] == 100

    # 3. Resume workflow with approval decision
    final_state = resume_human_review_workflow(
        thread_id=thread_id,
        approved=True,
        reviewer_name="Audit Manager",
        review_notes="Guidance cut confirmed.",
        checkpoint_db_path=checkpoint_db,
    )

    assert final_state.get("review_status") == "APPROVED"

    # 4. Verify DB persistence and credibility impact
    conn = get_connection(temp_db)
    c_row = conn.execute("SELECT * FROM contradictions WHERE statement_a_id=?", (stmt_a_id,)).fetchone()
    conn.close()

    assert c_row is not None
    assert c_row["contradiction_type"] == "HARD"
    assert c_row["review_status"] == "APPROVED"
    assert c_row["reviewed"] == 1

    # Credibility score should now be penalized (-20 points -> 80)
    assert scorer.score_executive(exec_id)["credibility_score"] == 80


def test_checkpoint_persistence_across_process_restart(temp_db: Path, checkpoint_db: Path):
    """Verify that interrupted state in SQLite checkpointer survives restart and resumes seamlessly."""
    comp_id = upsert_company("Reliance", "500325", "Conglomerate", db_path=temp_db)
    exec_id = upsert_executive("V. Srikanth", "CFO", comp_id, db_path=temp_db)

    stmt_a_id = insert_statement(
        executive_id=exec_id, company_id=comp_id, transcript_id=None, quarter="Q1FY24", year=2024,
        text="Our rural segment shows strong traction and is our primary growth driver.", db_path=temp_db
    )
    stmt_b_id = insert_statement(
        executive_id=exec_id, company_id=comp_id, transcript_id=None, quarter="Q2FY24", year=2024,
        text="We are now focusing on urban premium going forward.", db_path=temp_db
    )

    # 1. First execution -> interrupt at review queue
    state1 = run_hard_contradiction_workflow(
        statement_a_id=stmt_a_id, statement_b_id=stmt_b_id, cosine_similarity=0.72,
        db_path=temp_db, checkpoint_db_path=checkpoint_db
    )
    thread_id = generate_thread_id(stmt_a_id, stmt_b_id)
    assert state1.get("review_status") == "PENDING"

    # 2. Resume with REJECT decision
    state2 = resume_human_review_workflow(
        thread_id=thread_id, approved=False, reviewer_name="Analyst B",
        review_notes="Not a direct contradiction.", checkpoint_db_path=checkpoint_db
    )

    assert state2.get("review_status") == "REJECTED"

    # Credibility score must remain 100 after rejection
    scorer = CredibilityScorer(db_path=temp_db)
    assert scorer.score_executive(exec_id)["credibility_score"] == 100
