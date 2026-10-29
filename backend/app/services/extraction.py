"""Extract plain text from stored PDF and TXT files.

Document bytes are treated only as data. Nothing in a file is executed or
interpreted as application instructions. Scanned/image-only PDFs need OCR,
which is out of scope for this phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ExtractionError(Exception):
    """Base class for text extraction failures."""


class EmptyDocumentError(ExtractionError):
    """The file opened but contained no extractable text."""


class UnreadableDocumentError(ExtractionError):
    """The file could not be parsed as PDF or UTF-8 text."""


@dataclass(frozen=True)
class PageSpan:
    start: int
    end: int
    page_number: int | None


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    page_spans: list[PageSpan]


def extract_document(path: Path, content_type: str) -> ExtractedDocument:
    suffix = path.suffix.lower()
    if content_type == "application/pdf" or suffix == ".pdf":
        return _extract_pdf(path)
    if content_type == "text/plain" or suffix == ".txt":
        return _extract_txt(path)
    raise UnreadableDocumentError("Unsupported document type")


def _extract_txt(path: Path) -> ExtractedDocument:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UnreadableDocumentError("Text file is not valid UTF-8") from exc
    except OSError as exc:
        raise UnreadableDocumentError("Unable to read document") from exc
    if not text.strip():
        raise EmptyDocumentError("Document contains no extractable text")
    return ExtractedDocument(text=text, page_spans=[PageSpan(0, len(text), None)])


def _extract_pdf(path: Path) -> ExtractedDocument:
    try:
        reader = PdfReader(str(path))
        pages = list(reader.pages)
    except (PdfReadError, OSError, ValueError) as exc:
        raise UnreadableDocumentError("Unable to read PDF") from exc

    if not pages:
        raise EmptyDocumentError("Document contains no extractable text")

    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    for index, page in enumerate(pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise UnreadableDocumentError("Unable to extract text from PDF") from exc
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(page_text)
        cursor += len(page_text)
        spans.append(PageSpan(start=start, end=cursor, page_number=index))

    text = "".join(parts)
    if not text.strip():
        raise EmptyDocumentError("Document contains no extractable text")
    return ExtractedDocument(text=text, page_spans=spans)
