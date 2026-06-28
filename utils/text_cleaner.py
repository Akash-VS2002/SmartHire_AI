"""
utils/text_cleaner.py
---------------------
Cleans raw resume / job-description text before embedding or LLM processing.
Preserves technical keywords and skill names while removing noise.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


# Characters to collapse into a single space
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")

# Collapse 3+ consecutive newlines into 2 (keep paragraph breaks)
_MULTILINE_PATTERN = re.compile(r"\n{3,}")

# Remove non-printable control characters (except \n and \t)
_CONTROL_CHARS = re.compile(r"[^\x09\x0A\x20-\x7E]")

# Common PDF artefacts: page numbers, header/footer markers
_PAGE_MARKER = re.compile(r"(page\s*\d+\s*(of\s*\d+)?)", re.IGNORECASE)
_URL_PATTERN  = re.compile(r"https?://\S+|www\.\S+")


def clean_text(raw: str, remove_urls: bool = False) -> str:
    """
    Clean raw text extracted from a PDF or pasted text block.

    Steps:
        1. Remove non-printable control chars
        2. Optionally strip URLs
        3. Remove PDF page-number artefacts
        4. Normalise whitespace (spaces/tabs)
        5. Collapse excess blank lines

    Args:
        raw:         Raw input string.
        remove_urls: If True, strip http/www URLs.

    Returns:
        Cleaned string.
    """
    if not raw:
        return ""

    text = _CONTROL_CHARS.sub(" ", raw)

    if remove_urls:
        text = _URL_PATTERN.sub("", text)

    # Remove "Page 1 of 5"-style artefacts
    text = _PAGE_MARKER.sub("", text)

    # Normalise horizontal whitespace
    text = _WHITESPACE_PATTERN.sub(" ", text)

    # Collapse 3+ newlines → 2
    text = _MULTILINE_PATTERN.sub("\n\n", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks suitable for embedding.

    Args:
        text:       Input text to split.
        chunk_size: Target number of characters per chunk.
        overlap:    Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move forward by (chunk_size - overlap) to create sliding window
        start += chunk_size - overlap

    logger.debug(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={overlap}).")
    return chunks


def extract_candidate_name_heuristic(text: str) -> str:
    """
    Cheap heuristic to guess the candidate name from the top of a resume
    before the LLM agent runs (used as a fallback label only).

    Looks for the first non-empty line that:
      - Has 2-4 words
      - Is Title-Case or ALL-CAPS
      - Does not contain common header keywords

    Args:
        text: Cleaned resume text.

    Returns:
        Guessed name string, or "Unknown Candidate".
    """
    skip_keywords = {
        "resume", "curriculum", "vitae", "cv", "profile",
        "summary", "objective", "contact", "email", "phone",
    }

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if 2 <= len(words) <= 4:
            lower_words = [w.lower() for w in words]
            if any(kw in lower_words for kw in skip_keywords):
                continue
            # Accept Title-Case or ALL-CAPS lines as likely names
            if line.istitle() or line.isupper():
                return line.title()
    return "Unknown Candidate"
