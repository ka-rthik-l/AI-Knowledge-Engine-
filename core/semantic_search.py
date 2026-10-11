"""Semantic search engine using dense vector embeddings and cosine similarity."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from core.embeddings import embed
from core.vector_store import save_store, load_store
from config import TOP_K, THRESHOLD, STORE_PREFIX


class SemanticSearch:
    """Manages text chunk embeddings and provides cosine-similarity retrieval."""

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.vectors: List[np.ndarray] = []

    def add(self, texts: List[str], source: Optional[str] = None) -> None:
        """Embeds text chunks and adds them to memory.

        Args:
            texts: List of text strings to embed and index.
            source: Document source identifier or filename.
        """
        if not texts:
            return

        embeddings = embed(texts)

        for t, e in zip(texts, embeddings):
            self.entries.append({
                "text": t,
                "source": source,
            })
            self.vectors.append(np.array(e, dtype=np.float32))

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = THRESHOLD,
        debug: bool = False,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Searches indexed entries for chunks semantically similar to the query.

        Args:
            query: The user search or question string.
            top_k: Maximum number of ranked results to return.
            threshold: Minimum cosine similarity score threshold.
            debug: If True, prints all ranked candidate scores to stdout.

        Returns:
            List of tuples (entry_dict, score) sorted by score descending.
        """
        if not query or not query.strip() or not self.vectors:
            return []

        query_vec = np.array(embed([query])[0], dtype=np.float32)
        norm_query = float(np.linalg.norm(query_vec))
        if norm_query == 0.0:
            return []

        scores: List[float] = []

        for vec in self.vectors:
            norm_vec = float(np.linalg.norm(vec))
            if norm_vec == 0.0:
                scores.append(0.0)
            else:
                score = float(np.dot(query_vec, vec) / (norm_query * norm_vec))
                scores.append(score)

        ranked = sorted(
            zip(self.entries, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if debug:
            for entry, score in ranked:
                print(f"Source: {entry['source']}")
                print(f"Score: {score:.3f}")
                print(entry["text"])
                print("-" * 40)

        filtered = [r for r in ranked if r[1] >= threshold]

        return filtered[:top_k]

    def save(self, prefix: str = STORE_PREFIX) -> None:
        """Persists current entries and vectors to disk."""
        save_store(self.entries, self.vectors, prefix=prefix)

    def load(self, prefix: str = STORE_PREFIX) -> None:
        """Loads entries and vectors from disk into memory."""
        self.entries, loaded_vectors = load_store(prefix=prefix)
        self.vectors = [np.array(v, dtype=np.float32) for v in loaded_vectors]
