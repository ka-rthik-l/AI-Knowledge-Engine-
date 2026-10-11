"""Text embedding generation using sentence-transformers."""

from typing import Any, List, Optional
import numpy as np

_model: Optional[Any] = None
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Lazily loads and caches the SentenceTransformer model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        The loaded SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """Takes a list of strings and returns their dense vector embeddings.

    Args:
        texts: List of text strings to embed.

    Returns:
        Numpy array of embeddings for each input string.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_embedding_model()
    return np.array(model.encode(texts), dtype=np.float32)
