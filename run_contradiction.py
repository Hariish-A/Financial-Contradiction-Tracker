"""
run_contradiction.py
--------------------
Orchestrates the full Contradiction Engine (Milestones 3 & 4):

  1. Backfill missing statement embeddings in SQLite database.
  2. Test the NLI model on predefined contradiction pairs.
  3. Conduct semantic query searches across executive statements.
  4. Run the full contradiction detection pipeline (Milestone 4):
     - HARD contradictions via FAISS + DeBERTa NLI cross-encoder
     - SOFT contradictions via topic similarity + sentiment flip + hedge escalation
     - OMISSION contradictions via spaCy topic dropout detection

Usage:
  # Backfill embeddings for all statements
  python run_contradiction.py --backfill

  # Run NLI scorer on test cases
  python run_contradiction.py --test-cases

  # Search similar statements for an executive
  python run_contradiction.py --exec-id 1 --query "We are confident about revenue growth"

  # Run the full Milestone 4 detection pipeline
  python run_contradiction.py --run-pipeline

  # Run pipeline for a single executive only (useful for testing)
  python run_contradiction.py --run-pipeline --exec-id 1
"""

import sys
import argparse
import sqlite3
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import numpy as np

ROOT = Path(__file__).resolve().parent
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
    get_contradictions,
)
from contradiction.embeddings import compute_embeddings, StatementIndex
from contradiction.nli_scorer import score_contradiction
from contradiction.soft_detector import score_soft_contradiction
from contradiction.omission_detector import detect_omissions


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 3 helpers (backfill / test-cases / search)
# ─────────────────────────────────────────────────────────────────────────────

def backfill_embeddings(batch_size: int = 128):
    """Fetch all statements without embeddings and compute them in batches."""
    logger.info("Connecting to database to check for statements without embeddings...")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM statements WHERE embedding IS NULL")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logger.info("No statements with missing embeddings found. Everything is up-to-date!")
        return

    logger.info(f"Found {len(rows)} statements with missing embeddings. Starting backfill on CPU...")

    for i in tqdm(range(0, len(rows), batch_size), desc="Backfilling Embeddings"):
        batch = rows[i:i+batch_size]
        batch_ids = [r[0] for r in batch]
        batch_texts = [r[1] for r in batch]

        try:
            embeddings = compute_embeddings(batch_texts, batch_size=batch_size, show_progress=False)
            for stmt_id, emb in zip(batch_ids, embeddings):
                update_statement_embedding(stmt_id, emb)
        except Exception as e:
            logger.error(f"Error computing or saving embeddings for batch {i}-{i+batch_size}: {e}")
            sys.exit(1)

    logger.info("Embedding backfill completed successfully!")


def run_test_cases():
    """Verify NLI scorer on standard sample test cases from note.md."""
    logger.info("Running NLI model on test cases from note.md...")

    test_cases = [
        {
            "name": "Hard contradiction",
            "a": "We expect 18% revenue growth in the next quarter.",
            "b": "We are revising our guidance to 8% for the quarter."
        },
        {
            "name": "Soft contradiction (sentiment flip)",
            "a": "Our rural segment shows strong traction and is our primary growth driver.",
            "b": "We are now focusing on urban premium going forward."
        },
        {
            "name": "Soft contradiction (hedge escalation)",
            "a": "We are confident about margin expansion in H2.",
            "b": "Significant macro headwinds will suppress margin improvement."
        },
        {
            "name": "Non-contradictory (entailment)",
            "a": "We expect to hire more engineers in the coming months.",
            "b": "We will be expanding our technical workforce."
        }
    ]

    print("\n" + "="*80)
    print("NLI CONTRAST SCORING TEST RUN")
    print("="*80)

    for tc in test_cases:
        print(f"\nTest Case: {tc['name']}")
        print(f"  Statement A: '{tc['a']}'")
        print(f"  Statement B: '{tc['b']}'")

        try:
            res = score_contradiction(tc["a"], tc["b"])
            print("  Results:")
            print(f"    Contradiction Probability : {res['contradiction_score']:.4f}")
            print(f"    Neutral Probability       : {res['neutral_score']:.4f}")
            print(f"    Entailment Probability    : {res['entailment_score']:.4f}")
            print(f"    Verdict                   : {res['verdict'].upper()}")
        except Exception as e:
            logger.error(f"Failed to run test case '{tc['name']}': {e}")

    print("="*80 + "\n")


def search_executive_statements(exec_id: int, query: str, top_k: int = 5):
    """Retrieve semantically similar statements for a specific executive."""
    logger.info(f"Querying similar statements for executive ID {exec_id}...")

    conn = get_connection()
    exec_row = conn.execute(
        "SELECT name, role FROM executives WHERE id = ?", (exec_id,)
    ).fetchone()
    conn.close()

    if not exec_row:
        logger.error(f"Executive with ID {exec_id} not found in database.")
        return

    print(f"\nExecutive: {exec_row['name']} ({exec_row['role']})")
    print(f"Query text: '{query}'")

    index = StatementIndex(exec_id)
    results = index.retrieve_similar(query, top_k=top_k)

    if not results:
        print("No matches found (ensure embeddings are backfilled!).")
        return

    print("\nTop Matches:")
    print("-" * 80)
    for stmt, score in results:
        print(f"Score: {score:.4f} | Quarter: {stmt['quarter']} FY{stmt['year']}")
        print(f"Text : {stmt['text']}")
        print("-" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# Milestone 4: Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _get_existing_pairs(conn) -> set:
    """Return a set of (statement_a_id, statement_b_id, contradiction_type) already in DB."""
    rows = conn.execute(
        "SELECT statement_a_id, statement_b_id, contradiction_type FROM contradictions"
    ).fetchall()
    return {(r[0], r[1], r[2]) for r in rows}


def _run_pipeline_for_executive(
    exec_id: int,
    exec_name: str,
    exec_role: str,
    existing_pairs: set,
) -> dict:
    """
    Run the full HARD + SOFT contradiction scan for one executive.
    Returns a dict of counts for reporting.
    """
    counts = {"hard": 0, "soft": 0, "skipped_duplicates": 0}

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, text, quarter, year, sentiment, sentiment_score, embedding
        FROM statements
        WHERE executive_id = ?
        ORDER BY year ASC, quarter ASC
        """,
        (exec_id,),
    ).fetchall()
    conn.close()

    if len(rows) < 2:
        return counts

    statements = [dict(r) for r in rows]

    # Build FAISS index for this executive
    faiss_index = StatementIndex(exec_id)

    # Iterate statements chronologically, compare each to all PRIOR statements
    for i, stmt_b in enumerate(statements):
        if stmt_b.get("embedding") is None:
            continue

        # Retrieve top-K semantically similar PRIOR statements using FAISS
        similar_results = faiss_index.retrieve_similar(stmt_b["text"], top_k=10)

        for prior_stmt, cosine_sim in similar_results:
            stmt_a = prior_stmt

            # Skip if stmt_a is from the same or later quarter than stmt_b
            a_key = _quarter_sort_key(stmt_a["quarter"], stmt_a["year"])
            b_key = _quarter_sort_key(stmt_b["quarter"], stmt_b["year"])
            if a_key >= b_key:
                continue

            # Skip if topic similarity is too low (not the same subject)
            if cosine_sim < TOPIC_SIMILARITY_THRESHOLD:
                continue

            # ── HARD contradiction check ──────────────────────────────────
            hard_pair_key = (stmt_a["id"], stmt_b["id"], "HARD")
            if hard_pair_key not in existing_pairs:
                try:
                    nli_result = score_contradiction(stmt_a["text"], stmt_b["text"])
                    if nli_result["contradiction_score"] > HARD_CONTRADICTION_THRESHOLD:
                        details = {
                            "nli_contradiction_score": nli_result["contradiction_score"],
                            "nli_neutral_score":       nli_result["neutral_score"],
                            "nli_entailment_score":    nli_result["entailment_score"],
                            "cosine_similarity":       round(cosine_sim, 4),
                            "quarter_a": stmt_a["quarter"],
                            "quarter_b": stmt_b["quarter"],
                        }
                        insert_contradiction(
                            statement_a_id=stmt_a["id"],
                            statement_b_id=stmt_b["id"],
                            contradiction_type="HARD",
                            score=round(nli_result["contradiction_score"], 4),
                            details=details,
                        )
                        existing_pairs.add(hard_pair_key)
                        counts["hard"] += 1
                except Exception as e:
                    logger.warning(f"NLI scoring failed for pair ({stmt_a['id']}, {stmt_b['id']}): {e}")
            else:
                counts["skipped_duplicates"] += 1

            # ── SOFT contradiction check ──────────────────────────────────
            soft_pair_key = (stmt_a["id"], stmt_b["id"], "SOFT")
            if soft_pair_key not in existing_pairs:
                try:
                    soft_result = score_soft_contradiction(stmt_a, stmt_b)
                    if soft_result["is_soft_contradiction"]:
                        details = {
                            "topic_similarity":  soft_result["topic_similarity"],
                            "sentiment_flip":    soft_result["sentiment_flip"],
                            "hedge_escalation":  soft_result["hedge_escalation"],
                            "composite_score":   soft_result["composite_score"],
                            "cosine_similarity": round(cosine_sim, 4),
                            "quarter_a":         stmt_a["quarter"],
                            "quarter_b":         stmt_b["quarter"],
                        }
                        insert_contradiction(
                            statement_a_id=stmt_a["id"],
                            statement_b_id=stmt_b["id"],
                            contradiction_type="SOFT",
                            score=round(soft_result["composite_score"], 4),
                            details=details,
                        )
                        existing_pairs.add(soft_pair_key)
                        counts["soft"] += 1
                except Exception as e:
                    logger.warning(f"Soft scoring failed for pair ({stmt_a['id']}, {stmt_b['id']}): {e}")
            else:
                counts["skipped_duplicates"] += 1

    return counts


def _quarter_sort_key(quarter: str, year: int) -> tuple:
    """Returns (year, quarter_num) for chronological ordering."""
    import re
    m = re.match(r"Q(\d)", quarter)
    q_num = int(m.group(1)) if m else 0
    return (year, q_num)


def _run_omissions_for_executive(
    exec_id: int,
    existing_pairs: set,
) -> int:
    """Run omission detection for one executive. Returns count of omissions inserted."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, text, quarter, year FROM statements WHERE executive_id = ? ORDER BY year, quarter",
        (exec_id,),
    ).fetchall()
    conn.close()

    statements = [dict(r) for r in rows]
    omissions = detect_omissions(statements)

    inserted = 0
    for omission in omissions:
        key = (omission["statement_a_id"], omission["statement_b_id"], "OMISSION")
        if key in existing_pairs:
            continue
        try:
            insert_contradiction(
                statement_a_id=omission["statement_a_id"],
                statement_b_id=omission["statement_b_id"],
                contradiction_type="OMISSION",
                score=omission["score"],
                details=omission["details"],
            )
            existing_pairs.add(key)
            inserted += 1
        except Exception as e:
            logger.warning(f"Failed to insert omission: {e}")

    return inserted


def run_full_pipeline(exec_id_filter: int = None):
    """
    Full Milestone 4 contradiction detection pipeline.
    Scans all executives (or a single one if exec_id_filter is set).
    """
    logger.info("=" * 70)
    logger.info("MILESTONE 4 — FULL CONTRADICTION DETECTION PIPELINE")
    logger.info("=" * 70)

    # Step 1: Ensure all embeddings are present
    logger.info("Step 1/3 — Checking embeddings (auto-backfilling if needed)...")
    backfill_embeddings()

    # Fetch executives
    conn = get_connection()
    if exec_id_filter:
        executives = conn.execute(
            "SELECT id, name, role FROM executives WHERE id = ?", (exec_id_filter,)
        ).fetchall()
    else:
        executives = conn.execute(
            "SELECT id, name, role FROM executives ORDER BY id"
        ).fetchall()
    conn.close()

    if not executives:
        logger.error("No executives found in database. Run ingestion + extraction first.")
        return

    logger.info(f"Found {len(executives)} executive(s) to scan.")

    # Load existing pairs once to avoid repeated DB queries
    conn = get_connection()
    existing_pairs = _get_existing_pairs(conn)
    conn.close()

    # Step 2: HARD + SOFT detection per executive
    logger.info("Step 2/3 — Scanning for HARD and SOFT contradictions...")
    total_hard = total_soft = total_skipped = 0

    for exec_row in tqdm(executives, desc="Executives (HARD+SOFT)"):
        exec_id   = exec_row["id"]
        exec_name = exec_row["name"]
        exec_role = exec_row["role"]

        logger.debug(f"  Scanning: {exec_name} ({exec_role}) [id={exec_id}]")
        counts = _run_pipeline_for_executive(exec_id, exec_name, exec_role, existing_pairs)
        total_hard    += counts["hard"]
        total_soft    += counts["soft"]
        total_skipped += counts["skipped_duplicates"]

    # Step 3: OMISSION detection per executive
    logger.info("Step 3/3 — Scanning for OMISSION contradictions...")
    total_omissions = 0

    for exec_row in tqdm(executives, desc="Executives (OMISSION)"):
        omission_count = _run_omissions_for_executive(exec_row["id"], existing_pairs)
        total_omissions += omission_count

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE — Summary")
    logger.info(f"  HARD contradictions inserted  : {total_hard}")
    logger.info(f"  SOFT contradictions inserted  : {total_soft}")
    logger.info(f"  OMISSION contradictions       : {total_omissions}")
    logger.info(f"  Duplicate pairs skipped       : {total_skipped}")
    logger.info("=" * 70)

    # Quick DB verification
    conn = get_connection()
    breakdown = conn.execute(
        "SELECT contradiction_type, COUNT(*) as cnt FROM contradictions GROUP BY contradiction_type"
    ).fetchall()
    conn.close()

    logger.info("\nContradictions table totals:")
    for row in breakdown:
        logger.info(f"  {row['contradiction_type']:<10}: {row['cnt']} records")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Financial Contradiction Tracker — Contradiction Engine (Milestones 3 & 4)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill embeddings for all statements in the database"
    )
    group.add_argument(
        "--test-cases",
        action="store_true",
        help="Run NLI scorer on verification test cases"
    )
    group.add_argument(
        "--exec-id",
        type=int,
        metavar="ID",
        help="Run search query on statements of executive with this ID"
    )
    group.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Run the full Milestone 4 contradiction detection pipeline"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Search query text (required if --exec-id is provided)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search results to return (default: 5)"
    )
    parser.add_argument(
        "--filter-exec",
        type=int,
        metavar="ID",
        help="When used with --run-pipeline, limit scan to a single executive ID"
    )

    args = parser.parse_args()

    if args.exec_id is not None and not args.query:
        parser.error("--query is required when using --exec-id")

    if args.backfill:
        backfill_embeddings()
    elif args.test_cases:
        run_test_cases()
    elif args.exec_id is not None:
        search_executive_statements(args.exec_id, args.query, args.top_k)
    elif args.run_pipeline:
        run_full_pipeline(exec_id_filter=args.filter_exec)


if __name__ == "__main__":
    main()
