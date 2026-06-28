"""
utils/pdf_reader.py
--------------------
Handles PDF text extraction using PyPDF (pypdf).
Gracefully handles:
  - Empty PDFs
  - Corrupt / non-PDF files
  - Very large resumes (page-by-page streaming)
"""

import logging
from pathlib import Path
from typing import Union
import io

logger = logging.getLogger(__name__)


def extract_text_from_pdf(source: Union[str, Path, bytes, io.BytesIO]) -> str:
    """
    Extract full text from a PDF file.

    Args:
        source: Path to the PDF file, raw bytes, or a BytesIO object.

    Returns:
        Extracted text as a single string, or empty string on failure.
    """
    try:
        from pypdf import PdfReader  # pypdf >= 3.x

        # Normalise source into a file-like object
        if isinstance(source, (str, Path)):
            reader = PdfReader(str(source))
        elif isinstance(source, bytes):
            reader = PdfReader(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            reader = PdfReader(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        if len(reader.pages) == 0:
            logger.warning("PDF has no pages — returning empty string.")
            return ""

        pages_text: list[str] = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                pages_text.append(text)
            except Exception as e:
                logger.warning(f"Could not extract page {page_num}: {e}")
                continue

        full_text = "\n".join(pages_text).strip()

        if not full_text:
            logger.warning("PDF contained no extractable text (possibly scanned image).")
            return ""

        logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages.")
        return full_text

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """
    Convenience wrapper for Streamlit's UploadedFile objects.

    Args:
        uploaded_file: streamlit.runtime.uploaded_file_manager.UploadedFile

    Returns:
        Extracted text string.
    """
    try:
        bytes_data = uploaded_file.read()
        if not bytes_data:
            logger.warning(f"Uploaded file '{uploaded_file.name}' is empty.")
            return ""
        return extract_text_from_pdf(bytes_data)
    except Exception as e:
        logger.error(f"Failed to process uploaded file: {e}")
        return ""
