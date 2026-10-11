"""Persistent vector and metadata storage utilities."""

import os
import json
from typing import Any, List, Tuple, Union
import numpy as np
from config import STORE_PREFIX


def save_store(
    texts: List[Any],
    vectors: Union[List[Any], np.ndarray],
    prefix: str = STORE_PREFIX,
) -> None:
    """Saves embedding vectors and text entries to disk.

    Args:
        texts: List of text chunk metadata dictionaries.
        vectors: Vector embeddings corresponding to each chunk.
        prefix: File path prefix for the saved files.
    """
    directory = os.path.dirname(prefix)
    if directory:
        os.makedirs(directory, exist_ok=True)

    vectors_path = prefix + "_vectors.npy"
    texts_path = prefix + "_texts.json"

    np.save(vectors_path, np.array(vectors))

    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)


def load_store(prefix: str = STORE_PREFIX) -> Tuple[List[Any], np.ndarray]:
    """Loads embedding vectors and text entries from disk.

    Args:
        prefix: File path prefix for the stored files.

    Returns:
        A tuple of (texts, vectors) where texts is a list of entries and vectors
        is a numpy ndarray.

    Raises:
        FileNotFoundError: If vectors or texts file does not exist.
    """
    vectors_path = prefix + "_vectors.npy"
    texts_path = prefix + "_texts.json"

    if not os.path.exists(vectors_path) or not os.path.exists(texts_path):
        raise FileNotFoundError(
            f"Vector store files not found for prefix '{prefix}'. "
            f"Expected '{vectors_path}' and '{texts_path}'."
        )

    vectors = np.load(vectors_path)

    with open(texts_path, "r", encoding="utf-8") as f:
        texts = json.load(f)

    return texts, vectors
