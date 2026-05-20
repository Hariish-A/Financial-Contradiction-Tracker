"""
contradiction/nli_scorer.py
---------------------------
Calculates the NLI (Natural Language Inference) relationship between two statements.
Specifically, it scores the probability of "contradiction", "neutral", and "entailment".
Uses the cross-encoder/nli-deberta-v3-base model.
"""

import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import NLI_MODEL

# Global cached model and tokenizer
_nli_model: Optional[AutoModelForSequenceClassification] = None
_nli_tokenizer: Optional[AutoTokenizer] = None

def get_nli_model_and_tokenizer() -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Get or load the NLI model and tokenizer (singleton)."""
    global _nli_model, _nli_tokenizer
    if _nli_model is None or _nli_tokenizer is None:
        logger.info(f"Loading NLI model and tokenizer: {NLI_MODEL}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
        logger.info(f"NLI model loaded successfully on device: {device}.")
    return _nli_model, _nli_tokenizer


def score_contradiction(statement_a: str, statement_b: str) -> Dict[str, Any]:
    """
    Compute NLI probabilities for the pair (statement_a, statement_b).
    Returns a dict with:
      - contradiction_score: float
      - neutral_score: float
      - entailment_score: float
      - verdict: "contradiction" | "neutral" | "entailment"
    """
    model, tokenizer = get_nli_model_and_tokenizer()
    device = next(model.parameters()).device
    
    # Tokenize input pair
    inputs = tokenizer(
        statement_a,
        statement_b,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        
    probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    
    # Handle single element list if batch size squeeze behavior happens
    if not isinstance(probs, list):
        probs = [probs]
        
    # Map index to correct label
    id2label = getattr(model.config, "id2label", None)
    label_map = {}
    if id2label:
        for idx, label_name in id2label.items():
            name_lower = label_name.lower()
            if "contradict" in name_lower:
                label_map[idx] = "contradiction"
            elif "entail" in name_lower:
                label_map[idx] = "entailment"
            elif "neutr" in name_lower:
                label_map[idx] = "neutral"
            else:
                label_map[idx] = name_lower
    else:
        # Fallback standard NLI deberta order: 0=contradiction, 1=entailment, 2=neutral
        label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
        
    # Build standard scores dict
    scores = {
        "contradiction_score": 0.0,
        "entailment_score": 0.0,
        "neutral_score": 0.0,
        "verdict": "neutral"
    }
    
    for idx, prob in enumerate(probs):
        label_name = label_map.get(idx, f"label_{idx}")
        if label_name == "contradiction":
            scores["contradiction_score"] = float(prob)
        elif label_name == "entailment":
            scores["entailment_score"] = float(prob)
        elif label_name == "neutral":
            scores["neutral_score"] = float(prob)
            
    # Verdict is the label with the highest probability
    max_idx = int(np.argmax(probs))
    scores["verdict"] = label_map.get(max_idx, "neutral")
    
    return scores
