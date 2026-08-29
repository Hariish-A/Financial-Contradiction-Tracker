"""
tests/test_migrations.py
-------------------------
Tests for storage/migrations.py and review_status credibility impact.
"""

import sqlite3
import pytest
from pathlib import Path

from storage.database import (
    init_db,
    get_connection,
    upsert_company,
    upsert_executive,
    insert_contradiction,
)
from storage.migrations import run_migrations
from credibility.scorer import CredibilityScorer


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "migration_test.db"
    init_db(db_file)
    return db_file


def test_migrations_idempotency(temp_db: Path):
    """Verify that run_migrations can be called multiple times without errors."""
    run_migrations(temp_db)
    run_migrations(temp_db)

    conn = get_connection(temp_db)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(contradictions)")
    cols = {row["name"] for row in cursor.fetchall()}
    
    assert "review_status" in cols
    assert "reviewer_name" in cols
    assert "decision_source" in cols
    assert "llm_verdict" in cols
    assert "graph_thread_id" in cols

    # Verify financial_actuals table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='financial_actuals'")
    table = cursor.fetchone()
    assert table is not None
    conn.close()


def test_credibility_only_counts_approved_hard_rows(temp_db: Path, monkeypatch):
    """Verify that pending and rejected HARD contradictions do NOT penalize credibility score."""
    conn = get_connection(temp_db)
    # Setup company & executive
    comp_id = upsert_company("Test Company", "111111", "Finance", db_path=temp_db)
    exec_id = upsert_executive("Alice Smith", "CFO", comp_id, db_path=temp_db)

    # Insert 2 dummy statements
    cur = conn.execute(
        "INSERT INTO statements (executive_id, company_id, quarter, year, text) VALUES (?, ?, 'Q1FY24', 2024, 'Text A')",
        (exec_id, comp_id),
    )
    stmt_a = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO statements (executive_id, company_id, quarter, year, text) VALUES (?, ?, 'Q2FY24', 2024, 'Text B')",
        (exec_id, comp_id),
    )
    stmt_b = cur.lastrowid
    conn.commit()

    # 1. PENDING HARD contradiction -> score should remain 100
    conn.execute(
        """
        INSERT INTO contradictions (statement_a_id, statement_b_id, contradiction_type, score, details, review_status)
        VALUES (?, ?, 'HARD', 0.85, '{}', 'PENDING')
        """,
        (stmt_a, stmt_b),
    )
    conn.commit()
    conn.close()

    scorer = CredibilityScorer(db_path=temp_db)
    res = scorer.score_executive(exec_id)
    assert res["credibility_score"] == 100
    assert res["hard_contradictions"] == 0

    # 2. Update to APPROVED -> score should drop by 20 (to 80)
    conn = get_connection(temp_db)
    conn.execute("UPDATE contradictions SET review_status = 'APPROVED' WHERE statement_a_id = ?", (stmt_a,))
    conn.commit()
    conn.close()

    res = scorer.score_executive(exec_id)
    assert res["credibility_score"] == 80
    assert res["hard_contradictions"] == 1

    # 3. Update to REJECTED -> score should restore to 100
    conn = get_connection(temp_db)
    conn.execute("UPDATE contradictions SET review_status = 'REJECTED' WHERE statement_a_id = ?", (stmt_a,))
    conn.commit()
    conn.close()

    res = scorer.score_executive(exec_id)
    assert res["credibility_score"] == 100
    assert res["hard_contradictions"] == 0
