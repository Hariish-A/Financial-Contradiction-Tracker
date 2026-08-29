"""
contradiction/services.py
-------------------------
Reusable domain service layer for the Contradiction Engine (Phase 0 refactoring).

Separates candidate pair generation, NLI scoring/HARD adjudication, SOFT detection,
OMISSION detection, and DB persistence into modular, testable functions with optional
db_path support.
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    HARD_CONTRADICTION_THRESHOLD,
    SOFT_CONTRADICTION_THRESHOLD,
    TOPIC_SIMILARITY_THRESHOLD,
)
from storage.database import (
    get_connection,
    update_statement_embedding,
    insert_contradiction,
    DB_PATH,
)
from contradiction.embeddings import compute_embeddings, StatementIndex
from contradiction.nli_scorer import score_contradiction
from contradiction.soft_detector import score_soft_contradiction
from contradiction.omission_detector import detect_omissions


def _quarter_sort_key(quarter: str, year: int) -> tuple:
    """Returns (year, quarter_num) for chronological ordering."""
    m = re.match(r"Q(\d)", quarter, re.IGNORECASE)
    q_num = int(m.group(1)) if m else 0
    return (year, q_num)


def backfill_embeddings_service(batch_size: int = 128, db_path: Path = DB_PATH) -> int:
    """
    Fetch all statements without embeddings and compute them in batches.
    Returns the number of statements updated.
    """
    logger.info(f"Checking missing embeddings in DB: {db_path}")
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM statements WHERE embedding IS NULL")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logger.info("No statements with missing embeddings found.")
        return 0

    logger.info(f"Found {len(rows)} statements with missing embeddings. Starting backfill...")
    total_updated = 0

    for i in tqdm(range(0, len(rows), batch_size), desc="Backfilling Embeddings"):
        batch = rows[i : i + batch_size]
        batch_ids = [r[0] for r in batch]
        batch_texts = [r[1] for r in batch]

        try:
            embeddings = compute_embeddings(batch_texts, batch_size=batch_size, show_progress=False)
            conn = get_connection(db_path)
            for stmt_id, emb in zip(batch_ids, embeddings):
                emb_blob = emb.astype("float32").tobytes()
                conn.execute("UPDATE statements SET embedding=? WHERE id=?", (emb_blob, stmt_id))
            conn.commit()
            conn.close()
            total_updated += len(batch_ids)
        except Exception as e:
            logger.error(f"Error backfilling embeddings batch {i}-{i+batch_size}: {e}")
            raise e

    logger.info(f"Embedding backfill completed successfully. Updated {total_updated} statements.")
    return total_updated


def generate_candidate_pairs(
    exec_id: int,
    top_k: int = 10,
    db_path: Path = DB_PATH,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """
    Generate candidate statement pairs (stmt_a, stmt_b, cosine_similarity) for an executive.
    stmt_a is guaranteed to be chronologically PRIOR to stmt_b, and cosine_sim >= TOPIC_SIMILARITY_THRESHOLD.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT id, executive_id, company_id, transcript_id, quarter, year, text,
               statement_type, sentiment, sentiment_score, embedding
        FROM statements
        WHERE executive_id = ?
        ORDER BY year ASC, quarter ASC
        """,
        (exec_id,),
    ).fetchall()
    conn.close()

    if len(rows) < 2:
        return []

    statements = [dict(r) for r in rows]
    index = StatementIndex(exec_id)
    candidate_pairs = []

    for stmt_b in statements:
        if stmt_b.get("embedding") is None:
            continue

        similar_results = index.retrieve_similar(stmt_b["text"], top_k=top_k)

        for prior_stmt, cosine_sim in similar_results:
            stmt_a = prior_stmt

            # Skip same or later quarter
            a_key = _quarter_sort_key(stmt_a["quarter"], stmt_a["year"])
            b_key = _quarter_sort_key(stmt_b["quarter"], stmt_b["year"])
            if a_key >= b_key:
                continue

            # Topic similarity filter
            if cosine_sim < TOPIC_SIMILARITY_THRESHOLD:
                continue

            candidate_pairs.append((stmt_a, stmt_b, float(cosine_sim)))

    return candidate_pairs


def get_existing_pairs(db_path: Path = DB_PATH) -> Set[Tuple[int, int, str]]:
    """Return set of (statement_a_id, statement_b_id, contradiction_type) in DB."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT statement_a_id, statement_b_id, contradiction_type FROM contradictions"
    ).fetchall()
    conn.close()
    return {(r[0], r[1], r[2]) for r in rows}


def persist_contradiction_record(
    statement_a_id: int,
    statement_b_id: int,
    contradiction_type: str,
    score: float,
    details: dict,
    db_path: Path = DB_PATH,
    **extra_fields,
) -> int:
    """
    Persist a contradiction record into SQLite database with optional extra audit fields.
    """
    import json
    conn = get_connection(db_path)
    
    # Check extra fields dynamically if available
    columns = ["statement_a_id", "statement_b_id", "contradiction_type", "score", "details"]
    values = [statement_a_id, statement_b_id, contradiction_type, score, json.dumps(details)]

    for field, val in extra_fields.items():
        if val is not None:
            columns.append(field)
            values.append(json.dumps(val) if isinstance(val, (dict, list)) else val)

    col_names = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO contradictions ({col_names}) VALUES ({placeholders})"

    cur = conn.execute(sql, values)
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def run_executive_scan(
    exec_id: int,
    existing_pairs: Optional[Set[Tuple[int, int, str]]] = None,
    db_path: Path = DB_PATH,
    use_langgraph_for_hard: bool = False,
) -> Dict[str, int]:
    """
    Run full scan for one executive using candidate generation, HARD/SOFT, and OMISSION detectors.
    Returns counts dict.
    """
    if existing_pairs is None:
        existing_pairs = get_existing_pairs(db_path)

    counts = {
        "hard": 0,
        "soft": 0,
        "omissions": 0,
        "skipped_duplicates": 0,
        "auto_dismissed": 0,
        "llm_judged": 0,
        "pending_review": 0,
        "approved": 0,
        "rejected": 0,
        "failed": 0,
    }

    pairs = generate_candidate_pairs(exec_id, db_path=db_path)
    from orchestration.service import run_hard_contradiction_workflow

    for stmt_a, stmt_b, cosine_sim in pairs:
        # HARD contradiction check via LangGraph workflow
        hard_pair_key = (stmt_a["id"], stmt_b["id"], "HARD")
        if hard_pair_key not in existing_pairs:
            try:
                state = run_hard_contradiction_workflow(
                    statement_a_id=stmt_a["id"],
                    statement_b_id=stmt_b["id"],
                    cosine_similarity=cosine_sim,
                    db_path=db_path,
                )
                existing_pairs.add(hard_pair_key)

                status = state.get("review_status", "")
                if status == "APPROVED":
                    counts["hard"] += 1
                    counts["approved"] += 1
                elif status == "PENDING":
                    counts["hard"] += 1
                    counts["pending_review"] += 1
                elif status == "REJECTED":
                    counts["rejected"] += 1
                else:
                    counts["auto_dismissed"] += 1

                if state.get("decision_source") == "LLM":
                    counts["llm_judged"] += 1
            except Exception as e:
                logger.warning(f"LangGraph HARD workflow failed for pair ({stmt_a['id']}, {stmt_b['id']}): {e}")
                counts["failed"] += 1
        else:
            counts["skipped_duplicates"] += 1

        # SOFT contradiction check
        soft_pair_key = (stmt_a["id"], stmt_b["id"], "SOFT")
        if soft_pair_key not in existing_pairs:
            try:
                soft_res = score_soft_contradiction(stmt_a, stmt_b)
                if soft_res["is_soft_contradiction"]:
                    details = {
                        "topic_similarity": soft_res["topic_similarity"],
                        "sentiment_flip": soft_res["sentiment_flip"],
                        "hedge_escalation": soft_res["hedge_escalation"],
                        "composite_score": soft_res["composite_score"],
                        "cosine_similarity": round(cosine_sim, 4),
                        "quarter_a": stmt_a["quarter"],
                        "quarter_b": stmt_b["quarter"],
                    }
                    persist_contradiction_record(
                        statement_a_id=stmt_a["id"],
                        statement_b_id=stmt_b["id"],
                        contradiction_type="SOFT",
                        score=round(soft_res["composite_score"], 4),
                        details=details,
                        db_path=db_path,
                    )
                    existing_pairs.add(soft_pair_key)
                    counts["soft"] += 1
            except Exception as e:
                logger.warning(f"Soft scoring failed for pair ({stmt_a['id']}, {stmt_b['id']}): {e}")
        else:
            counts["skipped_duplicates"] += 1

    # OMISSION check
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, text, quarter, year FROM statements WHERE executive_id = ? ORDER BY year, quarter",
        (exec_id,),
    ).fetchall()
    conn.close()

    stmts_for_omission = [dict(r) for r in rows]
    omissions = detect_omissions(stmts_for_omission)

    for omission in omissions:
        key = (omission["statement_a_id"], omission["statement_b_id"], "OMISSION")
        if key in existing_pairs:
            continue
        try:
            persist_contradiction_record(
                statement_a_id=omission["statement_a_id"],
                statement_b_id=omission["statement_b_id"],
                contradiction_type="OMISSION",
                score=omission["score"],
                details=omission["details"],
                db_path=db_path,
            )
            existing_pairs.add(key)
            counts["omissions"] += 1
        except Exception as e:
            logger.warning(f"Failed to insert omission: {e}")

    return counts
