"""
tests/test_verification.py
---------------------------
Unit and integration tests for Phase 6 financial actuals repository, metric mapping,
and prediction verification workflow.
"""

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
from storage.financial_actuals import (
    upsert_financial_actual,
    get_financial_actuals,
    map_screener_metric,
)
from orchestration.verification_graph import run_prediction_verification
from credibility.scorer import CredibilityScorer


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "verification_test.db"
    init_db(db_file)
    return db_file


def test_metric_mapping_and_financial_actuals_repo(temp_db: Path):
    """Test Screener row metric mapping and financial actuals CRUD."""
    assert map_screener_metric("Sales") == "revenue_growth"
    assert map_screener_metric("OPM %") == "operating_margin"
    assert map_screener_metric("EPS in Rs") == "eps"

    comp_id = upsert_company("Wipro", "507685", "IT", db_path=temp_db)
    actual_id = upsert_financial_actual(
        company_id=comp_id,
        quarter="Q1FY24",
        metric="revenue_growth",
        value=14.5,
        db_path=temp_db,
    )
    assert actual_id > 0

    actuals = get_financial_actuals(comp_id, quarter="Q1FY24", metric="revenue_growth", db_path=temp_db)
    assert len(actuals) == 1
    assert actuals[0]["value"] == 14.5


def test_prediction_verification_workflow(temp_db: Path):
    """Test prediction verification workflow matching prediction with actual outcome."""
    comp_id = upsert_company("TCS", "532540", "IT", db_path=temp_db)
    exec_id = upsert_executive("K. Krithivasan", "CEO", comp_id, db_path=temp_db)

    stmt_id = insert_statement(
        executive_id=exec_id, company_id=comp_id, transcript_id=None,
        quarter="Q1FY24", year=2024, text="We target 12% revenue growth.",
        db_path=temp_db,
    )

    pred_id = insert_prediction(
        executive_id=exec_id, statement_id=stmt_id, quarter="Q1FY24",
        metric="revenue_growth", predicted_value=12.0, direction="up", db_path=temp_db,
    )

    # Insert actual
    upsert_financial_actual(
        company_id=comp_id, quarter="Q1FY24", metric="revenue_growth", value=14.0, db_path=temp_db
    )

    # Run verification graph
    res = run_prediction_verification(pred_id, db_path=temp_db)

    assert res["is_verified"] is True
    assert res["actual_value"] == 14.0
    assert res["is_direction_correct"] is True
    assert res["magnitude_error_pct"] == 16.67

    # Verify credibility scorer reward (+10 points for correct direction call)
    scorer = CredibilityScorer(db_path=temp_db)
    score_report = scorer.score_executive(exec_id)
    assert score_report["direction_correct"] == 1
    assert score_report["credibility_score"] == 100  # clamped to 100 (100 base + 10 reward clamped)
