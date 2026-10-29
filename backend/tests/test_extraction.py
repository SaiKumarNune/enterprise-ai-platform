from pathlib import Path

import pytest

from app.services.extraction import (
    EmptyDocumentError,
    UnreadableDocumentError,
    extract_document,
)
from tests.pdf_fixtures import pdf_with_pages


def test_extract_utf8_text_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("knowledge base notes", encoding="utf-8")

    extracted = extract_document(path, "text/plain")

    assert extracted.text == "knowledge base notes"
    assert extracted.page_spans[0].page_number is None


def test_extract_multipage_pdf_preserves_page_numbers(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf_with_pages("Hello World Page One", "Hello World Page Two"))

    extracted = extract_document(path, "application/pdf")

    assert "Hello World Page One" in extracted.text
    assert "Hello World Page Two" in extracted.text
    assert [span.page_number for span in extracted.page_spans] == [1, 2]


def test_extract_empty_text_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        extract_document(path, "text/plain")


def test_extract_unreadable_pdf_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a pdf structure")

    with pytest.raises((EmptyDocumentError, UnreadableDocumentError)):
        extract_document(path, "application/pdf")
