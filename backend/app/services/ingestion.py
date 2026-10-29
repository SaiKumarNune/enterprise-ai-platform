from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.services.chunking import chunk_text
from app.services.extraction import (
    EmptyDocumentError,
    ExtractionError,
    extract_document,
)
from app.services.storage import LocalStorage, StoredFileMissingError

PROCESSABLE_STATUSES = {"uploaded", "failed", "ready"}


class IngestionConflictError(Exception):
    """The document cannot be processed in its current state."""


def process_stored_document(
    db: Session,
    document: Document,
    storage: LocalStorage,
    *,
    chunk_size: int,
    overlap: int,
) -> Document:
    if document.status == "processing":
        raise IngestionConflictError("Document is already being processed")
    if document.status not in PROCESSABLE_STATUSES:
        raise IngestionConflictError("Document cannot be processed in its current state")
    if not document.storage_path:
        raise IngestionConflictError("Document has no stored file")

    try:
        path = storage.resolve(document.storage_path)
    except StoredFileMissingError:
        raise

    document.status = "processing"
    db.commit()

    try:
        extracted = extract_document(path, document.content_type)
        pieces = chunk_text(
            extracted.text,
            chunk_size=chunk_size,
            overlap=overlap,
            page_spans=extracted.page_spans,
        )
        if not pieces:
            raise EmptyDocumentError("Document contains no extractable text")

        db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        for piece in pieces:
            db.add(
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=piece.index,
                    content=piece.content,
                    page_number=piece.page_number,
                )
            )
        document.status = "ready"
        db.commit()
        db.refresh(document)
        return document
    except Exception:
        document_id = document.id
        db.rollback()
        failed = db.get(Document, document_id)
        if failed is not None:
            failed.status = "failed"
            db.commit()
        raise
