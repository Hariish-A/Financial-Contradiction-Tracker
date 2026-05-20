"""
storage/database.py
-------------------
SQLite schema initialisation + helper CRUD functions.
DuckDB is used for cross-quarter analytics queries (see analytics()).

Schema mirrors the design in note.md:
  companies → executives → statements → contradictions → predictions
"""

import sqlite3
import json
import numpy as np
import duckdb
from pathlib import Path
from loguru import logger
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "tracker.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# Schema DDL
# ─────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS companies (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT    NOT NULL,
    bse_code TEXT    NOT NULL UNIQUE,
    sector   TEXT
);

CREATE TABLE IF NOT EXISTS executives (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL,
    role       TEXT    NOT NULL,          -- CFO | CEO | MD
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    UNIQUE(name, company_id)
);

CREATE TABLE IF NOT EXISTS transcripts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id  INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    quarter     TEXT    NOT NULL,         -- e.g. Q1FY24
    year        INTEGER NOT NULL,
    source_url  TEXT,
    pdf_path    TEXT,
    raw_text    TEXT,
    processed   INTEGER DEFAULT 0,       -- 0=raw, 1=extracted
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS statements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    executive_id    INTEGER NOT NULL REFERENCES executives(id),
    company_id      INTEGER NOT NULL REFERENCES companies(id),
    transcript_id   INTEGER REFERENCES transcripts(id),
    quarter         TEXT    NOT NULL,
    year            INTEGER NOT NULL,
    text            TEXT    NOT NULL,
    statement_type  TEXT,                -- QUANTITATIVE_GUIDANCE | QUALITATIVE_GUIDANCE | HEDGED | DEFLECTION | FACTUAL_CLAIM
    sentiment       TEXT,                -- positive | negative | neutral
    sentiment_score REAL,
    embedding       BLOB,                -- serialised numpy float32 array
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS contradictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_a_id     INTEGER NOT NULL REFERENCES statements(id),
    statement_b_id     INTEGER NOT NULL REFERENCES statements(id),
    contradiction_type TEXT    NOT NULL,  -- HARD | SOFT | OMISSION
    score              REAL    NOT NULL,
    details            TEXT,             -- JSON blob
    reviewed           INTEGER DEFAULT 0,
    created_at         TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    executive_id    INTEGER NOT NULL REFERENCES executives(id),
    statement_id    INTEGER REFERENCES statements(id),
    quarter         TEXT    NOT NULL,
    metric          TEXT    NOT NULL,    -- revenue_growth | margin | ebitda | etc.
    predicted_value REAL,
    direction       TEXT,               -- up | down | stable
    actual_value    REAL,
    outcome_quarter TEXT,
    verified        INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_statements_exec   ON statements(executive_id);
CREATE INDEX IF NOT EXISTS idx_statements_quarter ON statements(quarter, year);
CREATE INDEX IF NOT EXISTS idx_contradictions_type ON contradictions(contradiction_type);
"""


# ─────────────────────────────────────────────────────────────────────
# Connection helpers
# ─────────────────────────────────────────────────────────────────────
def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they do not exist."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
    logger.info(f"Database initialised at {db_path}")


# ─────────────────────────────────────────────────────────────────────
# Company helpers
# ─────────────────────────────────────────────────────────────────────
def upsert_company(name: str, bse_code: str, sector: str = "") -> int:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO companies(name, bse_code, sector) VALUES(?,?,?)",
            (name, bse_code, sector),
        )
        row = conn.execute(
            "SELECT id FROM companies WHERE bse_code=?", (bse_code,)
        ).fetchone()
        return row["id"]


def get_company(bse_code: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM companies WHERE bse_code=?", (bse_code,)
        ).fetchone()


# ─────────────────────────────────────────────────────────────────────
# Executive helpers
# ─────────────────────────────────────────────────────────────────────
def upsert_executive(name: str, role: str, company_id: int) -> int:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO executives(name, role, company_id) VALUES(?,?,?)",
            (name, role, company_id),
        )
        row = conn.execute(
            "SELECT id FROM executives WHERE name=? AND company_id=?",
            (name, company_id),
        ).fetchone()
        return row["id"]


# ─────────────────────────────────────────────────────────────────────
# Transcript helpers
# ─────────────────────────────────────────────────────────────────────
def insert_transcript(
    company_id: int,
    quarter: str,
    year: int,
    source_url: str = "",
    pdf_path: str = "",
    raw_text: str = "",
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO transcripts(company_id, quarter, year, source_url, pdf_path, raw_text)
               VALUES(?,?,?,?,?,?)""",
            (company_id, quarter, year, source_url, pdf_path, raw_text),
        )
        return cur.lastrowid


def mark_transcript_processed(transcript_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE transcripts SET processed=1 WHERE id=?", (transcript_id,)
        )


def get_unprocessed_transcripts():
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM transcripts WHERE processed=0"
        ).fetchall()


# ─────────────────────────────────────────────────────────────────────
# Statement helpers
# ─────────────────────────────────────────────────────────────────────
def insert_statement(
    executive_id: int,
    company_id: int,
    transcript_id: int,
    quarter: str,
    year: int,
    text: str,
    statement_type: str = "",
    sentiment: str = "",
    sentiment_score: float = 0.0,
    embedding: Optional[np.ndarray] = None,
) -> int:
    emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO statements
               (executive_id, company_id, transcript_id, quarter, year,
                text, statement_type, sentiment, sentiment_score, embedding)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                executive_id, company_id, transcript_id, quarter, year,
                text, statement_type, sentiment, sentiment_score, emb_blob,
            ),
        )
        return cur.lastrowid


def load_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def update_statement_embedding(statement_id: int, embedding: np.ndarray) -> None:
    emb_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
    with get_connection() as conn:
        conn.execute(
            "UPDATE statements SET embedding=? WHERE id=?",
            (emb_blob, statement_id),
        )


def get_statements_for_executive(executive_id: int):
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM statements WHERE executive_id=? ORDER BY year, quarter",
            (executive_id,),
        ).fetchall()


# ─────────────────────────────────────────────────────────────────────
# Contradiction helpers
# ─────────────────────────────────────────────────────────────────────
def insert_contradiction(
    statement_a_id: int,
    statement_b_id: int,
    contradiction_type: str,
    score: float,
    details: dict,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO contradictions
               (statement_a_id, statement_b_id, contradiction_type, score, details)
               VALUES(?,?,?,?,?)""",
            (statement_a_id, statement_b_id, contradiction_type, score, json.dumps(details)),
        )
        return cur.lastrowid


def get_contradictions(contradiction_type: str = None):
    with get_connection() as conn:
        if contradiction_type:
            return conn.execute(
                "SELECT * FROM contradictions WHERE contradiction_type=? ORDER BY score DESC",
                (contradiction_type,),
            ).fetchall()
        return conn.execute(
            "SELECT * FROM contradictions ORDER BY score DESC"
        ).fetchall()


# ─────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────
def insert_prediction(
    executive_id: int,
    statement_id: int,
    quarter: str,
    metric: str,
    predicted_value: float = None,
    direction: str = "",
    outcome_quarter: str = "",
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (executive_id, statement_id, quarter, metric,
                predicted_value, direction, outcome_quarter)
               VALUES(?,?,?,?,?,?,?)""",
            (executive_id, statement_id, quarter, metric,
             predicted_value, direction, outcome_quarter),
        )
        return cur.lastrowid


# ─────────────────────────────────────────────────────────────────────
# DuckDB analytics
# ─────────────────────────────────────────────────────────────────────
def analytics(sql: str):
    """
    Run a read-only analytics query using DuckDB directly on the SQLite file.
    DuckDB can attach SQLite databases for fast OLAP-style queries.
    """
    con = duckdb.connect()
    con.execute(f"ATTACH '{DB_PATH}' AS tracker (TYPE sqlite, READ_ONLY)")
    return con.execute(sql).fetchdf()


# ─────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print(f"[OK] Database ready at {DB_PATH}")
