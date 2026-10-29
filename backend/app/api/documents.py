from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document
from app.schemas.document import DocumentCreate, DocumentResponse, DocumentUpdate
from app.services.storage import (
    FileTooLargeError,
    InvalidFileContentError,
    LocalStorage,
    StoredFileMissingError,
    UnsupportedFileTypeError,
    get_storage,
)

router = APIRouter()


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Document not found",
    )


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_document(
    payload: DocumentCreate,
    db: Session = Depends(get_db),
) -> Document:
    document = Document(
        filename=payload.filename,
        content_type=payload.content_type,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
) -> Document:
    stored = None
    try:
        stored = storage.save_upload(file)
        document = Document(
            filename=stored.filename,
            content_type=stored.content_type,
            storage_path=stored.storage_path,
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        return document
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="File exceeds the 10 MB upload limit",
        ) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF and plain-text files are supported",
        ) from exc
    except InvalidFileContentError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The uploaded file is not a valid PDF or UTF-8 text file",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        if stored is not None:
            storage.delete(stored.storage_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save document",
        ) from exc


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> list[Document]:
    statement = (
        select(Document)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(db.scalars(statement).all())


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: UUID,
    db: Session = Depends(get_db),
) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise _not_found()
    return document


@router.patch("/documents/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: UUID,
    payload: DocumentUpdate,
    db: Session = Depends(get_db),
) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise _not_found()
    updates = payload.model_dump(exclude_unset=True)
    for field_name, value in updates.items():
        setattr(document, field_name, value)
    db.commit()
    db.refresh(document)
    return document


@router.get("/documents/{document_id}/download")
def download_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
) -> FileResponse:
    document = db.get(Document, document_id)
    if document is None:
        raise _not_found()
    if not document.storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored file not found",
        )
    try:
        path = storage.resolve(document.storage_path)
    except StoredFileMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored file not found",
        ) from exc
    download_name = Path(document.filename).name or "download"
    ascii_name = "".join(
        char if char.isascii() and char not in '\\/"' else "_"
        for char in download_name
    ) or "download"
    return FileResponse(
        path=path,
        media_type=document.content_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_name}"; '
                f"filename*=UTF-8''{quote(download_name)}"
            )
        },
    )


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
) -> None:
    document = db.get(Document, document_id)
    if document is None:
        raise _not_found()
    storage_path = document.storage_path
    db.delete(document)
    db.commit()
    # Best-effort cleanup after a successful commit. If unlink fails, the
    # database row is already gone and an orphan file may remain on disk.
    storage.delete(storage_path)
