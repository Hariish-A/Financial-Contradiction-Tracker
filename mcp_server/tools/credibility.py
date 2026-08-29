"""
mcp_server/tools/credibility.py
-------------------------------
MCP tools for executive credibility score evaluation.
Uses CredibilityScorer domain service (never fictitious score tables).
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credibility.scorer import CredibilityScorer
from storage.database import DB_PATH


def get_credibility_score(
    executive_id: Optional[int] = None,
    executive_name: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Get executive credibility score(s).
    If executive_id or executive_name is supplied, scores that executive; otherwise scores all executives.
    """
    scorer = CredibilityScorer(db_path=db_path)

    if executive_id is not None:
        res = scorer.score_executive(executive_id)
        return [res] if res else []

    all_scores = scorer.score_all()

    if executive_name:
        name_lower = executive_name.lower()
        return [s for s in all_scores if name_lower in s["name"].lower()]

    return all_scores
