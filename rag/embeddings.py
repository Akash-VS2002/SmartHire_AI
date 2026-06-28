"""
rag/embeddings.py
-----------------
Wrapper around SentenceTransformer for producing text embeddings.
Used by the ChromaDB vector store for resume and JD chunks.
"""

import logging
from typing import List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Model name — "all-MiniLM-L6-v2" is small, fast, and highly accurate for
# semantic similarity tasks. Switch to "all-mpnet-base-v2" for higher quality.
DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """
    Load and cache the SentenceTransformer model.
    The lru_cache ensures the model is loaded only once per process.
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model '{model_name}': {e}")
        raise


def get_embeddings(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> List[List[float]]:
    """
    Generate dense vector embeddings for a list of text strings.

    Args:
        texts:      List of strings to embed.
        model_name: SentenceTransformer model identifier.
        batch_size: Number of texts to encode per batch (memory control).

    Returns:
        List of embedding vectors (each is a List[float]).
    """
    if not texts:
        return []

    model = _load_model(model_name)

    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # Convert numpy array rows → plain Python lists for ChromaDB
        return [emb.tolist() for emb in embeddings]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


def get_single_embedding(
    text: str,
    model_name: str = DEFAULT_MODEL,
) -> List[float]:
    """
    Generate an embedding for a single string.

    Args:
        text:       Input string.
        model_name: SentenceTransformer model identifier.

    Returns:
        Embedding vector as List[float].
    """
    result = get_embeddings([text], model_name=model_name)
    return result[0] if result else []


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Similarity score in [-1, 1]; returns 0.0 on dimension mismatch.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a ** 2 for a in vec_a) ** 0.5
    norm_b = sum(b ** 2 for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
