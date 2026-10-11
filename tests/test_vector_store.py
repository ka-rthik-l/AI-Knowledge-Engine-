import os
import shutil
import tempfile
import unittest
import numpy as np
from core.vector_store import save_store, load_store


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        prefix = os.path.join(self.test_dir, "test_store")
        sample_texts = [
            {"text": "Sample text with unicode: ©, é, π", "source": "doc1.pdf"},
            {"text": "Another chunk of text", "source": "doc2.pdf"},
        ]
        sample_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

        save_store(sample_texts, sample_vectors, prefix=prefix)
        loaded_texts, loaded_vectors = load_store(prefix=prefix)

        self.assertEqual(loaded_texts, sample_texts)
        np.testing.assert_array_almost_equal(loaded_vectors, sample_vectors)

    def test_auto_create_nested_directories(self):
        prefix = os.path.join(self.test_dir, "nested", "sub", "store")
        sample_texts = [{"text": "Hello nested", "source": "test.pdf"}]
        sample_vectors = np.array([[1.0, 2.0]], dtype=np.float32)

        save_store(sample_texts, sample_vectors, prefix=prefix)
        self.assertTrue(os.path.exists(prefix + "_vectors.npy"))
        self.assertTrue(os.path.exists(prefix + "_texts.json"))

    def test_load_nonexistent_store_raises_error(self):
        nonexistent_prefix = os.path.join(self.test_dir, "nonexistent")
        with self.assertRaises(FileNotFoundError):
            load_store(prefix=nonexistent_prefix)


if __name__ == "__main__":
    unittest.main()
