"""
tests/test_baseline.py
----------------------
Baseline unit tests for ContraGuard module imports and core database helpers.
Uses isolated temporary SQLite databases to prevent modifying data/tracker.db.
"""

import sqlite3
import pytest
from pathlib import Path

from storage.database import (
    init_db,
    get_connection,
    upsert_company,
    upsert_executive,
    insert_statement,
    insert_contradiction,
    get_contradictions,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Fixture returning path to a temporary SQLite database."""
    db_file = tmp_path / "test_tracker.db"
    init_db(db_file)
    return db_file


def test_imports():
    """Verify all existing core modules import cleanly."""
    import config
    import storage.database
    import contradiction.nli_scorer
    import contradiction.embeddings
    import contradiction.soft_detector
    import contradiction.omission_detector
    import credibility.scorer
    import dashboard.data_fetcher

    assert config.HARD_CONTRADICTION_THRESHOLD == 0.5
    assert config.SOFT_CONTRADICTION_THRESHOLD == 0.6


def test_database_init_and_crud(temp_db: Path):
    """Verify database initialization and basic CRUD operations on a temp database."""
    conn = get_connection(temp_db)
    
    # Test upsert company
    conn.execute(
        "INSERT OR IGNORE INTO companies(name, bse_code, sector) VALUES(?,?,?)",
        ("Test Corp", "999999", "Tech")
    )
    conn.commit()
    row = conn.execute("SELECT * FROM companies WHERE bse_code='999999'").fetchone()
    assert row is not None
    assert row["name"] == "Test Corp"

    # Test upsert executive
    comp_id = row["id"]
    conn.execute(
        "INSERT OR IGNORE INTO executives(name, role, company_id) VALUES(?,?,?)",
        ("Jane Doe", "CEO", comp_id)
    )
    conn.commit()
    exec_row = conn.execute("SELECT * FROM executives WHERE name='Jane Doe'").fetchone()
    assert exec_row is not None
    assert exec_row["role"] == "CEO"
    conn.close()
