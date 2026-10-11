"""Configuration settings for the AI Knowledge Engine.

Provides hyperparameters for text chunking, semantic retrieval,
vector store persistence, and generative answering.
"""

import os
from typing import Final

# Number of words per text chunk
CHUNK_SIZE: Final[int] = int(os.getenv("CHUNK_SIZE", "120"))

# Number of overlapping words between consecutive chunks
CHUNK_OVERLAP: Final[int] = int(os.getenv("CHUNK_OVERLAP", "30"))

# Maximum number of top relevant chunks to retrieve for context
TOP_K: Final[int] = int(os.getenv("TOP_K", "5"))

# Minimum cosine similarity threshold (range 0.0 to 1.0) for retrieval
THRESHOLD: Final[float] = float(os.getenv("THRESHOLD", "0.3"))

# Path prefix for storing vector embeddings and chunk metadata
STORE_PREFIX: Final[str] = os.getenv("STORE_PREFIX", os.path.join("storage", "store"))

# Default local LLM model for answer generation
LLM_MODEL: Final[str] = os.getenv("LLM_MODEL", "phi3")


def validate_config() -> None:
    """Validates configuration parameters to prevent runtime errors."""
    if CHUNK_SIZE <= 0:
        raise ValueError(f"CHUNK_SIZE must be positive, got {CHUNK_SIZE}")
    if CHUNK_OVERLAP < 0:
        raise ValueError(f"CHUNK_OVERLAP cannot be negative, got {CHUNK_OVERLAP}")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be strictly less than CHUNK_SIZE ({CHUNK_SIZE})"
        )
    if TOP_K <= 0:
        raise ValueError(f"TOP_K must be positive, got {TOP_K}")
    if not (0.0 <= THRESHOLD <= 1.0):
        raise ValueError(f"THRESHOLD must be between 0.0 and 1.0, got {THRESHOLD}")


validate_config()
