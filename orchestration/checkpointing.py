"""
orchestration/checkpointing.py
------------------------------
SQLite checkpointer initialization for LangGraph workflow state persistence.
Keeps graph state checkpointer separate from tracker.db at data/langgraph_checkpoints.db.
"""

import sqlite3
from pathlib import Path
from typing import Generator
from contextlib import contextmanager
from langgraph.checkpoint.sqlite import SqliteSaver

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DB_PATH = ROOT / "data" / "langgraph_checkpoints.db"
CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_checkpointer(db_path: Path = CHECKPOINT_DB_PATH):
    """
    Context manager yielding a thread-safe SqliteSaver checkpointer instance.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        saver.setup()
        yield saver
