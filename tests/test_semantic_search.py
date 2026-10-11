import unittest
from unittest.mock import patch
import numpy as np
from core.semantic_search import SemanticSearch


class TestSemanticSearch(unittest.TestCase):
    def setUp(self):
        self.engine = SemanticSearch()

    def test_search_empty_store_returns_empty(self):
        results = self.engine.search("test query")
        self.assertEqual(results, [])

    def test_search_empty_query_returns_empty(self):
        self.engine.entries = [{"text": "Hello", "source": "doc1.pdf"}]
        self.engine.vectors = [np.array([1.0, 0.0], dtype=np.float32)]

        self.assertEqual(self.engine.search(""), [])
        self.assertEqual(self.engine.search("   "), [])

    @patch("core.semantic_search.embed")
    def test_add_and_rank_similarity(self, mock_embed):
        # Mock embeddings: doc1=[1, 0], doc2=[0, 1], query=[1, 0]
        # Similarity with doc1 should be 1.0, doc2 should be 0.0
        def side_effect(texts):
            if texts == ["doc1 text", "doc2 text"]:
                return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            if texts == ["doc1 query"]:
                return np.array([[1.0, 0.0]], dtype=np.float32)
            return np.zeros((len(texts), 2), dtype=np.float32)

        mock_embed.side_effect = side_effect

        self.engine.add(["doc1 text", "doc2 text"], source="source.pdf")
        self.assertEqual(len(self.engine.entries), 2)
        self.assertEqual(len(self.engine.vectors), 2)

        results = self.engine.search("doc1 query", top_k=2, threshold=0.1)
        self.assertEqual(len(results), 1)
        top_entry, score = results[0]
        self.assertEqual(top_entry["text"], "doc1 text")
        self.assertAlmostEqual(score, 1.0, places=3)

    @patch("core.semantic_search.embed")
    def test_threshold_cutoff(self, mock_embed):
        mock_embed.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

        self.engine.entries = [
            {"text": "Relevant", "source": "a.pdf"},
            {"text": "Irrelevant", "source": "b.pdf"},
        ]
        # doc1 matches query (score 1.0), doc2 is perpendicular (score 0.0)
        self.engine.vectors = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]

        results = self.engine.search("query", threshold=0.5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["text"], "Relevant")

    @patch("core.semantic_search.embed")
    def test_top_k_limits(self, mock_embed):
        mock_embed.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

        self.engine.entries = [{"text": f"Doc {i}", "source": "a.pdf"} for i in range(5)]
        self.engine.vectors = [np.array([1.0, 0.0], dtype=np.float32) for _ in range(5)]

        results = self.engine.search("query", top_k=3, threshold=0.0)
        self.assertEqual(len(results), 3)

    @patch("core.semantic_search.embed")
    def test_zero_vector_safe_handling(self, mock_embed):
        mock_embed.return_value = np.array([[0.0, 0.0]], dtype=np.float32)

        self.engine.entries = [{"text": "Doc with zero vector", "source": "a.pdf"}]
        self.engine.vectors = [np.array([0.0, 0.0], dtype=np.float32)]

        results = self.engine.search("query", threshold=0.0)
        self.assertEqual(results, [])

    @patch("core.semantic_search.embed")
    def test_add_empty_list_does_nothing(self, mock_embed):
        self.engine.add([], source="empty.pdf")
        mock_embed.assert_not_called()
        self.assertEqual(len(self.engine.entries), 0)


if __name__ == "__main__":
    unittest.main()
