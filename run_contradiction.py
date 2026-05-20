"""
run_contradiction.py
--------------------
Orchestrates Milestone 3:
1. Backfilling missing statement embeddings in SQLite database.
2. Testing the NLI model on predefined contradiction pairs.
3. Conducting semantic query searches across executive statements.

Usage:
  # Run backfill of embeddings for statements in DB
  python run_contradiction.py --backfill

  # Run test cases of NLI model on test pairs
  python run_contradiction.py --test-cases

  # Search similar statements for an executive
  python run_contradiction.py --exec-id 1 --query "We are confident about revenue growth"
"""

import sys
import argparse
import sqlite3
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from storage.database import get_connection, update_statement_embedding
from contradiction.embeddings import compute_embeddings, StatementIndex
from contradiction.nli_scorer import score_contradiction

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
    
    # Process in batches
    for i in tqdm(range(0, len(rows), batch_size), desc="Backfilling Embeddings"):
        batch = rows[i:i+batch_size]
        batch_ids = [r[0] for r in batch]
        batch_texts = [r[1] for r in batch]
        
        try:
            # Generate L2-normalized embeddings
            embeddings = compute_embeddings(batch_texts, batch_size=batch_size, show_progress=False)
            
            # Save to database
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
    
    # Check if executive exists
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


def main():
    parser = argparse.ArgumentParser(
        description="Financial Contradiction Tracker — Contradiction Engine (Milestone 3)"
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
    
    args = parser.parse_args()
    
    if args.exec_id is not None and not args.query:
        parser.error("--query is required when using --exec-id")
        
    if args.backfill:
        backfill_embeddings()
    elif args.test_cases:
        run_test_cases()
    elif args.exec_id is not None:
        search_executive_statements(args.exec_id, args.query, args.top_k)


if __name__ == "__main__":
    main()
