from app.services.chunking import chunk_text
from app.services.extraction import PageSpan


def test_chunking_returns_empty_list_for_blank_input() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n") == []


def test_chunking_keeps_short_text_as_one_chunk() -> None:
    chunks = chunk_text("hello world", chunk_size=1000, overlap=200)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].content == "hello world"
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len("hello world")


def test_chunking_splits_long_text_with_overlap() -> None:
    text = "abcdefghij" * 30
    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert len(chunks) > 1
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].content == text[:50]
    overlap = chunks[0].content[-10:]
    assert chunks[1].content.startswith(overlap)
    for chunk in chunks:
        assert chunk.content
        assert len(chunk.content) <= 50


def test_chunking_prefers_paragraph_boundaries() -> None:
    first = "A" * 40
    second = "B" * 40
    text = first + "\n\n" + second
    chunks = chunk_text(text, chunk_size=50, overlap=5)

    assert chunks[0].content == first
    assert second in chunks[1].content


def test_chunking_maps_page_numbers_from_spans() -> None:
    text = "page-one-text\n\npage-two-text"
    spans = [
        PageSpan(0, 13, 1),
        PageSpan(13, len(text), 2),
    ]
    chunks = chunk_text(text, chunk_size=20, overlap=0, page_spans=spans)

    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 2
