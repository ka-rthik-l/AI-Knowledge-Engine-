import unittest
from core.chunker import chunk_text


class TestChunker(unittest.TestCase):
    def test_chunk_text_normal(self):
        text = "word1 word2 word3 word4 word5"
        chunks = chunk_text(text, chunk_size=3, overlap=1)
        expected = ["word1 word2 word3", "word3 word4 word5", "word5"]
        self.assertEqual(chunks, expected)

    def test_chunk_text_no_overlap(self):
        text = "alpha beta gamma delta"
        chunks = chunk_text(text, chunk_size=2, overlap=0)
        expected = ["alpha beta", "gamma delta"]
        self.assertEqual(chunks, expected)

    def test_chunk_text_smaller_than_chunk_size(self):
        text = "just two words"
        chunks = chunk_text(text, chunk_size=5, overlap=2)
        self.assertEqual(chunks, ["just two words"])

    def test_chunk_text_empty_and_whitespace(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t  "), [])

    def test_chunk_text_invalid_input_types(self):
        self.assertEqual(chunk_text(None), [])  # type: ignore[arg-type]
        self.assertEqual(chunk_text(123), [])  # type: ignore[arg-type]

    def test_chunk_text_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            chunk_text("hello world", chunk_size=0, overlap=0)

        with self.assertRaises(ValueError):
            chunk_text("hello world", chunk_size=-5, overlap=0)

    def test_chunk_text_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("hello world", chunk_size=5, overlap=-1)

        with self.assertRaises(ValueError):
            chunk_text("hello world", chunk_size=5, overlap=5)

        with self.assertRaises(ValueError):
            chunk_text("hello world", chunk_size=5, overlap=6)


if __name__ == "__main__":
    unittest.main()
