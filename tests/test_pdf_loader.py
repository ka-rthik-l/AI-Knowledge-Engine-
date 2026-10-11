import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from ingestion.pdf_loader import load_pdf


class TestPdfLoader(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_nonexistent_pdf_raises_error(self):
        fake_path = os.path.join(self.test_dir, "nonexistent.pdf")
        with self.assertRaises(FileNotFoundError):
            load_pdf(fake_path)

    def test_zero_byte_pdf_returns_empty_string(self):
        empty_pdf_path = os.path.join(self.test_dir, "empty.pdf")
        with open(empty_pdf_path, "wb") as f:
            pass  # create 0-byte file

        result = load_pdf(empty_pdf_path)
        self.assertEqual(result, "")

    def test_missing_pypdf_raises_import_error(self):
        dummy_path = os.path.join(self.test_dir, "dummy.pdf")
        with open(dummy_path, "wb") as f:
            f.write(b"%PDF-1.4 header")

        with patch.dict(sys.modules, {"pypdf": None}):
            with self.assertRaises(ImportError):
                load_pdf(dummy_path)

    def test_corrupted_pdf_returns_empty_string(self):
        corrupt_pdf_path = os.path.join(self.test_dir, "corrupt.pdf")
        with open(corrupt_pdf_path, "wb") as f:
            f.write(b"not a valid pdf content")

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.side_effect = Exception("Corrupt file")

        with patch.dict(sys.modules, {"pypdf": mock_pypdf}):
            result = load_pdf(corrupt_pdf_path)
            self.assertEqual(result, "")

    def test_successful_pdf_extraction(self):
        valid_pdf_path = os.path.join(self.test_dir, "sample.pdf")
        with open(valid_pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 sample content")

        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "Page 1 text"
        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = "Page 2 text"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page_1, mock_page_2]

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.return_value = mock_reader

        with patch.dict(sys.modules, {"pypdf": mock_pypdf}):
            result = load_pdf(valid_pdf_path)
            self.assertIn("Page 1 text", result)
            self.assertIn("Page 2 text", result)


if __name__ == "__main__":
    unittest.main()
