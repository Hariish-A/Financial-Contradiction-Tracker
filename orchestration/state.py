"""
orchestration/state.py
----------------------
Typed dict definition for the LangGraph contradiction state graph.
"""

from typing import TypedDict, Optional, Dict, Any


class ContradictionState(TypedDict):
    # Identifiers & statement references
    statement_a_id: int
    statement_b_id: int
    statement_a_text: str
    statement_b_text: str

    # Executive & Company Metadata
    company_name: str
    executive_name: str
    executive_role: str
    quarter_a: str
    quarter_b: str
    year_a: int
    year_b: int

    # Retrieval & NLI metrics
    cosine_similarity: float
    nli_contradiction_score: float
    nli_neutral_score: float
    nli_entailment_score: float
    nli_verdict: str

    # Routing & escalation
    routing_decision: str  # AUTO_REJECT | AUTO_HARD | LLM_EVALUATE | HUMAN_REVIEW

    # LLM Judge fields
    llm_verdict: Optional[str]  # contradiction | consistent | uncertain
    llm_confidence: Optional[float]
    llm_explanation: Optional[str]
    llm_metadata: Optional[Dict[str, Any]]

    # Review status & decisions
    review_status: str  # NOT_REQUIRED | PENDING | APPROVED | REJECTED | LEGACY_APPROVED
    decision_source: str  # NLI | LLM | HUMAN | LEGACY
    reviewer_name: Optional[str]
    review_notes: Optional[str]
    reviewed_at: Optional[str]

    # Process & error metadata
    db_path: Optional[str]
    graph_thread_id: str
    workflow_version: str
    error_message: Optional[str]
    is_completed: bool
