"""
dashboard/data_fetcher.py
--------------------------
Cached data-fetching functions for the Streamlit dashboard.
All heavy DB reads are wrapped in @st.cache_data so the UI stays instant.
"""

import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from storage.database import (
    get_connection,
    get_all_executives,
    get_predictions_for_executive,
    get_contradictions_for_executive,
    get_statement_count_for_executive,
    update_prediction_actual,
)
from credibility.scorer import CredibilityScorer


# ─────────────────────────────────────────────────────────────────────────────
# Credibility scores
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_all_credibility_scores() -> list[dict]:
    scorer = CredibilityScorer()
    return scorer.score_all()


@st.cache_data(ttl=60)
def fetch_credibility_score(exec_id: int) -> dict:
    scorer = CredibilityScorer()
    return scorer.score_executive(exec_id)


# ─────────────────────────────────────────────────────────────────────────────
# Executives
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_executives() -> pd.DataFrame:
    rows = get_all_executives()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# ─────────────────────────────────────────────────────────────────────────────
# Contradictions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_contradictions_df(exec_id: Optional[int] = None) -> pd.DataFrame:
    """
    Return a DataFrame of contradictions enriched with statement text and quarter info.
    If exec_id is None, fetch for all executives.
    """
    conn = get_connection()

    if exec_id is not None:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.contradiction_type,
                c.score,
                c.details,
                sa.text   AS statement_a_text,
                sa.quarter AS quarter_a,
                sa.year   AS year_a,
                sb.text   AS statement_b_text,
                sb.quarter AS quarter_b,
                sb.year   AS year_b,
                e.name    AS executive_name,
                e.role    AS executive_role,
                co.name   AS company_name
            FROM contradictions c
            JOIN statements sa ON sa.id = c.statement_a_id
            JOIN statements sb ON sb.id = c.statement_b_id
            JOIN executives e  ON e.id  = sa.executive_id
            JOIN companies  co ON co.id = sa.company_id
            WHERE sa.executive_id = ?
            ORDER BY c.score DESC
            """,
            (exec_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT
                c.id,
                c.contradiction_type,
                c.score,
                c.details,
                sa.text   AS statement_a_text,
                sa.quarter AS quarter_a,
                sa.year   AS year_a,
                sb.text   AS statement_b_text,
                sb.quarter AS quarter_b,
                sb.year   AS year_b,
                e.name    AS executive_name,
                e.role    AS executive_role,
                co.name   AS company_name
            FROM contradictions c
            JOIN statements sa ON sa.id = c.statement_a_id
            JOIN statements sb ON sb.id = c.statement_b_id
            JOIN executives e  ON e.id  = sa.executive_id
            JOIN companies  co ON co.id = sa.company_id
            ORDER BY c.score DESC
            """
        ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        d = dict(row)
        try:
            d["details"] = json.loads(d["details"]) if d["details"] else {}
        except Exception:
            d["details"] = {}
        records.append(d)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_predictions_df(exec_id: Optional[int] = None) -> pd.DataFrame:
    conn = get_connection()
    if exec_id is not None:
        rows = conn.execute(
            """
            SELECT p.*, e.name AS executive_name, e.role, co.name AS company_name
            FROM predictions p
            JOIN executives e  ON e.id = p.executive_id
            JOIN companies  co ON co.id = e.company_id
            WHERE p.executive_id = ?
            ORDER BY p.quarter
            """,
            (exec_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT p.*, e.name AS executive_name, e.role, co.name AS company_name
            FROM predictions p
            JOIN executives e  ON e.id = p.executive_id
            JOIN companies  co ON co.id = e.company_id
            ORDER BY p.quarter
            """
        ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# ─────────────────────────────────────────────────────────────────────────────
# DB summary stats
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_summary_stats() -> dict:
    conn = get_connection()
    companies     = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    executives    = conn.execute("SELECT COUNT(*) FROM executives").fetchone()[0]
    statements    = conn.execute("SELECT COUNT(*) FROM statements").fetchone()[0]
    transcripts   = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
    hard_count    = conn.execute("SELECT COUNT(*) FROM contradictions WHERE contradiction_type='HARD'").fetchone()[0]
    soft_count    = conn.execute("SELECT COUNT(*) FROM contradictions WHERE contradiction_type='SOFT'").fetchone()[0]
    omit_count    = conn.execute("SELECT COUNT(*) FROM contradictions WHERE contradiction_type='OMISSION'").fetchone()[0]
    pred_count    = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    verified_count= conn.execute("SELECT COUNT(*) FROM predictions WHERE verified=1").fetchone()[0]
    conn.close()
    return {
        "companies":      companies,
        "executives":     executives,
        "statements":     statements,
        "transcripts":    transcripts,
        "hard":           hard_count,
        "soft":           soft_count,
        "omissions":      omit_count,
        "total_contradictions": hard_count + soft_count + omit_count,
        "predictions":    pred_count,
        "verified":       verified_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Semantic search (live, not cached — result depends on query string)
# ─────────────────────────────────────────────────────────────────────────────

def run_semantic_search(exec_id: int, query: str, top_k: int = 8) -> list[dict]:
    from contradiction.embeddings import StatementIndex
    index   = StatementIndex(exec_id)
    results = index.retrieve_similar(query, top_k=top_k)
    out = []
    for stmt, score in results:
        out.append({
            "score":   round(float(score), 4),
            "quarter": stmt.get("quarter", ""),
            "year":    stmt.get("year", ""),
            "type":    stmt.get("statement_type", ""),
            "sentiment": stmt.get("sentiment", ""),
            "text":    stmt.get("text", ""),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Write helpers (non-cached)
# ─────────────────────────────────────────────────────────────────────────────

def verify_prediction(pred_id: int, actual_value: float) -> None:
    update_prediction_actual(pred_id, actual_value, verified=1)
    # Invalidate all caches so the UI refreshes
    st.cache_data.clear()


@st.cache_data(ttl=10)
def fetch_pending_reviews() -> pd.DataFrame:
    """Fetch pending review candidates for the Streamlit Review Queue UI."""
    from orchestration.service import list_pending_reviews
    items = list_pending_reviews()
    if not items:
        return pd.DataFrame()
    return pd.DataFrame(items)


@st.cache_data(ttl=10)
def fetch_review_history() -> pd.DataFrame:
    """Fetch historical review decisions (APPROVED, REJECTED, LEGACY_APPROVED)."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT
            c.id AS contradiction_id,
            c.contradiction_type,
            c.score,
            c.review_status,
            c.reviewer_name,
            c.review_notes,
            c.reviewed_at,
            c.decision_source,
            c.llm_verdict,
            c.llm_confidence,
            c.llm_explanation,
            c.graph_thread_id,
            sa.text AS statement_a_text,
            sa.quarter AS quarter_a,
            sb.text AS statement_b_text,
            sb.quarter AS quarter_b,
            e.name AS executive_name,
            co.name AS company_name
        FROM contradictions c
        JOIN statements sa ON sa.id = c.statement_a_id
        JOIN statements sb ON sb.id = c.statement_b_id
        JOIN executives e ON e.id = sa.executive_id
        JOIN companies co ON co.id = sa.company_id
        WHERE c.review_status IN ('APPROVED', 'REJECTED', 'LEGACY_APPROVED')
        ORDER BY c.reviewed_at DESC, c.id DESC
        """
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def submit_review_decision(
    thread_id: str,
    contradiction_id: int,
    approved: bool,
    reviewer_name: str = "human",
    review_notes: str = "",
) -> None:
    """
    Resume LangGraph workflow or update review status for a pending contradiction candidate.
    Invalidates Streamlit cache upon completion.
    """
    from orchestration.service import resume_human_review_workflow
    import datetime

    if thread_id and thread_id.startswith("hard_"):
        try:
            resume_human_review_workflow(
                thread_id=thread_id,
                approved=approved,
                reviewer_name=reviewer_name,
                review_notes=review_notes,
            )
        except Exception:
            # Fallback direct DB update if graph thread is uncheckpointed
            status = "APPROVED" if approved else "REJECTED"
            reviewed = 1 if approved else 0
            conn = get_connection()
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
                (status, reviewer_name, review_notes, datetime.datetime.now().isoformat(), reviewed, contradiction_id),
            )
            conn.commit()
            conn.close()
    else:
        status = "APPROVED" if approved else "REJECTED"
        reviewed = 1 if approved else 0
        conn = get_connection()
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
            (status, reviewer_name, review_notes, datetime.datetime.now().isoformat(), reviewed, contradiction_id),
        )
        conn.commit()
        conn.close()

    st.cache_data.clear()
