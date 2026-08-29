"""
orchestration/routing.py
------------------------
Routing policies and threshold evaluation for the LangGraph contradiction workflow.
"""

# Configurable routing thresholds
NLI_AUTO_REJECT_MAX = 0.20
NLI_AUTO_HARD_MIN = 0.80
LLM_JUDGE_MIN_CONFIDENCE = 0.70


def route_nli_result(nli_contradiction_prob: float, nli_verdict: str) -> str:
    """
    Evaluate NLI score distribution against thresholds.
    Returns routing decision string: 'AUTO_HARD' | 'AUTO_REJECT' | 'LLM_EVALUATE'
    """
    if nli_contradiction_prob >= NLI_AUTO_HARD_MIN:
        return "AUTO_HARD"

    if nli_contradiction_prob <= NLI_AUTO_REJECT_MAX and nli_verdict != "contradiction":
        return "AUTO_REJECT"

    return "LLM_EVALUATE"


def route_llm_result(llm_verdict: str, confidence: float) -> str:
    """
    Evaluate LLM judge result against confidence thresholds.
    Returns routing decision string: 'HUMAN_REVIEW' | 'AUTO_REJECT'
    """
    if llm_verdict == "contradiction":
        return "HUMAN_REVIEW"

    if llm_verdict == "consistent" and confidence >= LLM_JUDGE_MIN_CONFIDENCE:
        return "AUTO_REJECT"

    # Verdict is uncertain, low confidence, or error fallback -> send to human review
    return "HUMAN_REVIEW"
