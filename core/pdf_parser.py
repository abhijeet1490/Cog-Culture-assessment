"""PDF parsing utilities using PyMuPDF (fitz)."""

from dataclasses import dataclass
from typing import List
import re
import pymupdf


@dataclass
class PageContent:
    """Represents extracted text from a single PDF page."""
    page_number: int
    text: str


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 30) -> List[PageContent]:
    """
    Extract clean, reading-order text page-by-page from a PDF byte stream.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.
        max_pages: Safety cap on maximum pages processed.

    Returns:
        List of PageContent objects with 1-indexed page numbers.
    """
    if not pdf_bytes:
        raise ValueError("Uploaded PDF file is empty.")

    pages: List[PageContent] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            raise ValueError("PDF contains zero pages.")

        limit = min(doc.page_count, max_pages)
        for idx in range(limit):
            page = doc[idx]
            raw_text = page.get_text("text", sort=True) or ""
            # Normalize excessive blank lines while preserving paragraph structure
            cleaned = re.sub(r"[ \t]+", " ", raw_text)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            if cleaned:
                pages.append(PageContent(page_number=idx + 1, text=cleaned))

    if not pages:
        raise ValueError(
            "No selectable text could be extracted from this PDF. "
            "Please upload a text-based PDF (scanned image-only PDFs are not supported)."
        )

    return pages


def format_pages_for_llm(pages: List[PageContent], max_chars: int = 45000) -> str:
    """Concatenate extracted pages with clear page markers for LLM claim extraction."""
    blocks = []
    total = 0
    for page in pages:
        block = f"--- PAGE {page.page_number} ---\n{page.text}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                blocks.append(block[:remaining] + "\n[Truncated]")
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)
