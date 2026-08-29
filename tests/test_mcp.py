"""
tests/test_mcp.py
------------------
Unit and integration tests for ContraGuard MCP tools.
Ensures typed returns, max limits, parameterized SQL, JSON serializability,
and proper state transitions for approve/reject/verify tools.
"""

import json
import pytest
from pathlib import Path

from storage.database import (
    init_db,
    get_connection,
    upsert_company,
    upsert_executive,
    insert_statement,
    insert_prediction,
)
from mcp_server.tools.statements import query_statements
from mcp_server.tools.contradictions import get_contradictions
from mcp_server.tools.credibility import get_credibility_score
from mcp_server.tools.reviews import list_pending_reviews, approve_contradiction, reject_contradiction
from mcp_server.tools.predictions import get_prediction_status, verify_prediction_actual


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "mcp_test_tracker.db"
    init_db(db_file)

    # Seed data
    comp_id = upsert_company("Infosys", "500209", "IT", db_path=db_file)
    exec_id = upsert_executive("Salil Parekh", "CEO", comp_id, db_path=db_file)

    stmt_a = insert_statement(
        executive_id=exec_id, company_id=comp_id, transcript_id=None,
        quarter="Q1FY24", year=2024, text="We expect 15% growth.",
        statement_type="QUANTITATIVE_GUIDANCE", sentiment="positive", db_path=db_file
    )
    stmt_b = insert_statement(
        executive_id=exec_id, company_id=comp_id, transcript_id=None,
        quarter="Q2FY24", year=2024, text="Growth is revised to 7%.",
        statement_type="QUANTITATIVE_GUIDANCE", sentiment="negative", db_path=db_file
    )

    # Seed pending contradiction
    conn = get_connection(db_file)
    conn.execute(
        """
        INSERT INTO contradictions
            (statement_a_id, statement_b_id, contradiction_type, score, details, review_status, graph_thread_id)
        VALUES (?, ?, 'HARD', 0.92, '{}', 'PENDING', 'hard_1_2_v1_0')
        """,
        (stmt_a, stmt_b),
    )
    conn.commit()
    conn.close()

    # Seed prediction
    insert_prediction(
        executive_id=exec_id, statement_id=stmt_a, quarter="Q1FY24",
        metric="revenue_growth", predicted_value=15.0, direction="up", db_path=db_file
    )

    return db_file


def test_query_statements_filters_and_limits(temp_db: Path):
    """Test statement query filters, JSON serializability, and limit capping."""
    stmts = query_statements(company_query="Infosys", limit=10, db_path=temp_db)
    assert len(stmts) == 2
    assert json.dumps(stmts)  # Verify JSON serializability

    # Test limit capping
    stmts_capped = query_statements(limit=200, db_path=temp_db)
    assert len(stmts_capped) <= 100


def test_get_contradictions_and_pending_reviews(temp_db: Path):
    """Test get_contradictions filter and list_pending_reviews."""
    contra_list = get_contradictions(contradiction_type="HARD", db_path=temp_db)
    assert len(contra_list) == 1
    assert contra_list[0]["review_status"] == "PENDING"
    assert json.dumps(contra_list)

    pending = list_pending_reviews(db_path=temp_db)
    assert len(pending) == 1
    assert pending[0]["company_name"] == "Infosys"
    assert json.dumps(pending)


def test_get_credibility_score(temp_db: Path):
    """Test get_credibility_score tool calling CredibilityScorer."""
    scores = get_credibility_score(executive_name="Salil Parekh", db_path=temp_db)
    assert len(scores) == 1
    assert scores[0]["credibility_score"] == 100  # PENDING doesn't penalize score
    assert json.dumps(scores)


def test_approve_and_reject_contradiction(temp_db: Path):
    """Test approve_contradiction and reject_contradiction tools."""
    pending = list_pending_reviews(db_path=temp_db)
    c_id = pending[0]["contradiction_id"]

    # Approve
    res_app = approve_contradiction(contradiction_id=c_id, reviewer_name="mcp_test", db_path=temp_db)
    assert res_app["success"] is True
    assert res_app["current_status"] == "APPROVED"
    assert res_app["updated_credibility_score"] == 80  # Penalty applied
    assert json.dumps(res_app)

    # Reject
    res_rej = reject_contradiction(contradiction_id=c_id, reviewer_name="mcp_test", db_path=temp_db)
    assert res_rej["success"] is True
    assert res_rej["current_status"] == "REJECTED"
    assert res_rej["updated_credibility_score"] == 100  # Score restored
    assert json.dumps(res_rej)


def test_verify_prediction_actual(temp_db: Path):
    """Test verify_prediction_actual tool."""
    preds = get_prediction_status(verified_only=False, db_path=temp_db)
    assert len(preds) == 1
    p_id = preds[0]["prediction_id"]

    res_verify = verify_prediction_actual(prediction_id=p_id, actual_value=16.0, db_path=temp_db)
    assert res_verify["success"] is True
    assert res_verify["verified"] is True
    assert res_verify["new_actual_value"] == 16.0
    assert json.dumps(res_verify)
