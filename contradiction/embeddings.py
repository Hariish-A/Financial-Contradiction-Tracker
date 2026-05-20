"""
contradiction/embeddings.py
----------------------------
Handles statement embedding generation via SentenceTransformer and FAISS vector index retrieval.
"""

import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from loguru import logger
from typing import Optional, List, Tuple, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import EMBEDDING_MODEL
from storage.database import get_statements_for_executive, load_embedding

# Global cached model
_embedding_model: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    """Get or load the SentenceTransformer model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
        # Since CUDA is not available, it defaults to CPU
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


def compute_embeddings(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Compute L2-normalized embeddings for a list of texts.
    Returns a numpy array of shape (len(texts), dimension).
    """
    if not texts:
        return np.empty((0, 768), dtype=np.float32)
    
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True
    )
    return np.array(embeddings, dtype=np.float32)


class StatementIndex:
    """
    An in-memory FAISS index containing statement embeddings for a specific executive.
    Allows retrieving semantically similar statements.
    """
    def __init__(self, executive_id: int):
        self.executive_id = executive_id
        self.statements: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.dimension = 768  # nickmuchi/finbert-tone-... produces 768-dim embeddings
        
        self._build_index()

    def _build_index(self) -> None:
        """Fetch all statements with embeddings for this executive and build the FAISS index."""
        raw_rows = get_statements_for_executive(self.executive_id)
        
        valid_statements = []
        embeddings_list = []
        
        for row in raw_rows:
            stmt_dict = dict(row)
            if stmt_dict.get("embedding") is not None:
                try:
                    emb = load_embedding(stmt_dict["embedding"])
                    if emb.shape[0] == self.dimension:
                        # Ensure L2-normalized for cosine similarity
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm
                        embeddings_list.append(emb)
                        valid_statements.append(stmt_dict)
                    else:
                        logger.warning(
                            f"Statement ID {stmt_dict['id']} has invalid embedding dimension: {emb.shape[0]} instead of {self.dimension}"
                        )
                except Exception as e:
                    logger.error(f"Error loading embedding for statement ID {stmt_dict['id']}: {e}")
        
        if not embeddings_list:
            logger.debug(f"No statements with embeddings found for executive_id {self.executive_id}.")
            return
            
        embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
        
        # Inner Product of L2-normalized vectors is Cosine Similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_matrix)
        self.statements = valid_statements
        
        logger.debug(
            f"Built FAISS index for executive_id {self.executive_id} with {len(self.statements)} statements."
        )

    def retrieve_similar(self, query_text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top_k semantically similar statements for this executive.
        Returns a list of tuples: (statement_dict, cosine_similarity_score).
        """
        if self.index is None or not self.statements:
            return []
            
        # Encode and normalize query
        query_emb = compute_embeddings([query_text], show_progress=False)
        
        # Search the index
        k = min(top_k, len(self.statements))
        scores, indices = self.index.search(query_emb, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.statements):
                continue
            results.append((self.statements[idx], float(score)))
            
        return results
