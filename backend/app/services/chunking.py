from __future__ import annotations

from dataclasses import dataclass

from app.services.extraction import PageSpan

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    page_number: int | None
    start_offset: int
    end_offset: int


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    page_spans: list[PageSpan] | None = None,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be zero or positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n")
    length = len(normalized)
    chunks: list[TextChunk] = []
    start = 0
    index = 0

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = _boundary_end(normalized, start, end)
        raw = normalized[start:end]
        stripped = raw.strip()
        if stripped:
            lead = len(raw) - len(raw.lstrip())
            abs_start = start + lead
            abs_end = abs_start + len(stripped)
            chunks.append(
                TextChunk(
                    index=index,
                    content=stripped,
                    page_number=_page_at(abs_start, page_spans),
                    start_offset=abs_start,
                    end_offset=abs_end,
                )
            )
            index += 1
        if end >= length:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def _boundary_end(text: str, start: int, end: int) -> int:
    window = text[start:end]
    min_pos = max(len(window) // 2, 1)
    for separator in ("\n\n", "\n", ". ", "? ", "! "):
        pos = window.rfind(separator)
        if pos >= min_pos:
            return start + pos + len(separator)
    return end


def _page_at(offset: int, page_spans: list[PageSpan] | None) -> int | None:
    if not page_spans:
        return None
    for span in page_spans:
        if span.start <= offset < span.end or (offset == span.end and span.start == span.end):
            return span.page_number
    if offset >= page_spans[-1].end:
        return page_spans[-1].page_number
    return page_spans[0].page_number
