"""
storage/migrations.py
---------------------
Additive, idempotent migration system for ContraGuard.

Extends SQLite schema with review audit fields on contradictions,
adds uniqueness constraints for pipeline idempotency, and creates
the financial_actuals table for prediction verification.
"""

import sqlite3
from pathlib import Path
from loguru import logger
from storage.database import get_connection, DB_PATH

MIGRATION_VERSION = 1


def run_migrations(db_path: Path = DB_PATH) -> None:
    """
    Run additive, idempotent schema migrations on the SQLite database.
    Safe to execute multiple times.
    """
    conn = get_connection(db_path)
    
    try:
        cursor = conn.cursor()

        # 1. Check existing columns in contradictions table
        cursor.execute("PRAGMA table_info(contradictions)")
        cols = {row["name"] for row in cursor.fetchall()}

        new_columns = {
            "review_status": "TEXT DEFAULT 'NOT_REQUIRED'",   # NOT_REQUIRED | PENDING | APPROVED | REJECTED | LEGACY_APPROVED
            "reviewer_name": "TEXT",
            "review_notes": "TEXT",
            "reviewed_at": "TEXT",
            "decision_source": "TEXT DEFAULT 'NLI'",          # NLI | LLM | HUMAN | LEGACY
            "nli_scores_json": "TEXT",
            "llm_verdict": "TEXT",
            "llm_confidence": "REAL",
            "llm_explanation": "TEXT",
            "llm_metadata_json": "TEXT",
            "graph_thread_id": "TEXT",
            "workflow_version": "TEXT DEFAULT '1.0'",
        }

        for col_name, col_def in new_columns.items():
            if col_name not in cols:
                logger.info(f"Adding column '{col_name}' to contradictions table in {db_path}")
                cursor.execute(f"ALTER TABLE contradictions ADD COLUMN {col_name} {col_def}")

        # 2. Backfill historical HARD contradictions as LEGACY_APPROVED if review_status is default/unset
        cursor.execute(
            """
            UPDATE contradictions
            SET review_status = 'LEGACY_APPROVED',
                decision_source = 'LEGACY',
                reviewed = 1
            WHERE contradiction_type = 'HARD'
              AND (review_status IS NULL OR review_status = 'NOT_REQUIRED')
              AND (reviewed = 1 OR reviewed = 0)
            """
        )

        # 3. Ensure SOFT and OMISSION have review_status = 'NOT_REQUIRED' if null
        cursor.execute(
            """
            UPDATE contradictions
            SET review_status = 'NOT_REQUIRED',
                decision_source = 'NLI'
            WHERE contradiction_type IN ('SOFT', 'OMISSION')
              AND review_status IS NULL
            """
        )

        # 4. Create uniqueness constraint index on (statement_a_id, statement_b_id, contradiction_type)
        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_contradictions_unique_pair
            ON contradictions(statement_a_id, statement_b_id, contradiction_type)
            """
        )

        # 5. Create financial_actuals table for Phase 6 metrics verification
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS financial_actuals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id    INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
                quarter       TEXT    NOT NULL,
                metric        TEXT    NOT NULL,
                value         REAL    NOT NULL,
                unit_currency TEXT    DEFAULT 'INR Crores',
                source_url    TEXT,
                fetched_at    TEXT    DEFAULT (datetime('now')),
                provenance    TEXT,
                UNIQUE(company_id, quarter, metric)
            )
            """
        )

        # Index for fast lookup on company/quarter/metric
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_actuals_comp_qtr ON financial_actuals(company_id, quarter)"
        )

        conn.commit()
        logger.info(f"Idempotent database migrations completed successfully for {db_path}.")

    except Exception as exc:
        conn.rollback()
        logger.error(f"Migration error on {db_path}: {exc}")
        raise exc
    finally:
        conn.close()


if __name__ == "__main__":
    run_migrations()
