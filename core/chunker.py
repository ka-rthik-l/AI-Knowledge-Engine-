"""Text chunking utilities for splitting documents into overlapping windows."""

from typing import List
from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Splits a string of text into overlapping word chunks.

    Args:
        text: The input document text to chunk.
        chunk_size: Number of words included in each chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        A list of string chunks. Returns empty list if input is empty or contains no words.

    Raises:
        ValueError: If chunk_size <= 0 or overlap < 0 or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap cannot be negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    if not text or not isinstance(text, str):
        return []

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: List[str] = []

    for i in range(0, len(words), step):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))

    return chunks
