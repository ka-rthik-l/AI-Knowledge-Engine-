"""PDF document loader for text extraction."""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_pdf(path: str) -> str:
    """Extracts text from a PDF file page-by-page.

    Args:
        path: Filesystem path to the PDF file.

    Returns:
        Extracted plain text content, or an empty string if extraction fails.

    Raises:
        FileNotFoundError: If the specified PDF path does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF file not found: '{path}'")

    if os.path.getsize(path) == 0:
        logger.warning("PDF file '%s' is empty (0 bytes).", path)
        return ""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "The 'pypdf' library is required to load PDF files. "
            "Please install it with 'pip install pypdf'."
        ) from exc

    text = ""
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as exc:
        logger.warning("Failed to parse PDF '%s': %s", path, exc)
        return ""

    return text
