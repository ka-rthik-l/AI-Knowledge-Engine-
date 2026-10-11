import numpy as np
from core.embeddings import embed
from core.vector_store import save_store, load_store
from config import TOP_K, THRESHOLD

class SemanticSearch:
    def __init__(self):
        self.entries = []
        self.vectors = []

    def add(self, texts, source=None):
        embeddings = embed(texts)

        for t, e in zip(texts, embeddings):
            self.entries.append({
                "text": t,
                "source": source
            })
            self.vectors.append(e)


    def search(self, query, top_k=TOP_K, threshold=THRESHOLD, debug=False):
        query_vec = embed([query])[0]

        scores = []

        for vec in self.vectors:
            score = np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec)
            )
            scores.append(score)

        ranked = sorted(zip(self.entries, scores),
                        key=lambda x: x[1],
                        reverse=True)

        if debug:
            for entry, score in ranked:
                print(f"Source: {entry['source']}")
                print(f"Score: {score:.3f}")
                print(entry["text"])
                print("-" * 40)


        filtered = [r for r in ranked if r[1] >= threshold]

        return filtered[:top_k]


    def save(self):
        save_store(self.entries, self.vectors)
    def load(self):
        self.entries, self.vectors = load_store()