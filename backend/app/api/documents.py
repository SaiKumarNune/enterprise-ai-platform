from pathlib import Path
from typing import Annotated
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models.acl import DocumentACL
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.models.user import User
from app.schemas.chunk import DocumentChunkResponse
from app.schemas.document import DocumentCreate, DocumentResponse, DocumentUpdate
from app.schemas.processing_job import ProcessingJobResponse
from app.schemas.permission import PermissionGrantRequest, PermissionGrantResponse
from app.services.audit import record_audit
from app.services.authorization import (
    DocumentAccessDenied,
    Principal,
    authorized_document_clause,
    has_privilege,
    require_document,
)
from app.services.embeddings import EmbeddingBackend, EmbeddingError, get_embedder
from app.services.extraction import EmptyDocumentError, ExtractionError
from app.services.ingestion import IngestionConflictError, process_stored_document
from app.services.processing_jobs import JobEnqueueError, enqueue_processing_job, get_job
from app.services.permissions import PermissionError, grant_privilege, list_grants, revoke_privilege
from app.services.storage import (
    FileTooLargeError,
    InvalidFileContentError,
    LocalStorage,
    StoredFileMissingError,
    UnsupportedFileTypeError,
    get_storage,
)

router = APIRouter()
CurrentUser = Annotated[User, Depends(get_current_user)]


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Document not found",
    )


def _principal(user: User) -> Principal:
    return Principal.from_user(user)


def _authorized_document(
    db: Session,
    principal: Principal,
    document_id: UUID,
    minimum: str,
) -> Document:
    try:
        return require_document(db, principal, document_id, minimum)
    except DocumentAccessDenied:
        existing = db.get(Document, document_id)
        if existing is not None:
            record_audit(
                db,
                action="document.access_denied",
                outcome="denied",
                principal=principal,
                resource_type="document",
                resource_id=document_id,
                details={"privilege": minimum},
            )
            db.commit()
        raise _not_found() from None


@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_document(
    payload: DocumentCreate,
    user: CurrentUser,
    db: Session = Depends(get_db),
) -> Document:
    principal = _principal(user)
    document = Document(
        filename=payload.filename,
        content_type=payload.content_type,
        tenant_id=principal.tenant_id,
        owner_id=principal.user_id,
    )
    db.add(document)
    record_audit(
        db,
        action="document.create",
        outcome="success",
        principal=principal,
        resource_type="document",
        resource_id=document.id,
    )
    db.commit()
    db.refresh(document)
    return document


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    user: CurrentUser,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
) -> Document:
    principal = _principal(user)
    stored = None
    try:
        stored = storage.save_upload(file)
        document = Document(
            filename=stored.filename,
            content_type=stored.content_type,
            storage_path=stored.storage_path,
            tenant_id=principal.tenant_id,
            owner_id=principal.user_id,
        )
        db.add(document)
        record_audit(
            db,
            action="document.upload",
            outcome="success",
            principal=principal,
            resource_type="document",
            resource_id=document.id,
        )
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
    user: CurrentUser,
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> list[Document]:
    principal = _principal(user)
    statement = (
        select(Document)
        .where(authorized_document_clause(principal, "read"))
        .order_by(Document.created_at.desc(), Document.id.desc())
        .offset(skip)
        .limit(limit)
    )
    if hasattr(db, "documents"):
        db.last_statement = statement
        rows = [item for item in db.documents if has_privilege(db, principal, item, "read")]
        rows.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return rows[skip : skip + limit]
    return list(db.scalars(statement).all())


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: UUID,
    user: CurrentUser,
    db: Session = Depends(get_db),
) -> Document:
    return _authorized_document(db, _principal(user), document_id, "read")


@router.patch("/documents/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: UUID,
    payload: DocumentUpdate,
    user: CurrentUser,
    db: Session = Depends(get_db),
) -> Document:
    document = _authorized_document(db, _principal(user), document_id, "write")
    updates = payload.model_dump(exclude_unset=True)
    for field_name, value in updates.items():
        setattr(document, field_name, value)
    db.commit()
    db.refresh(document)
    return document


@router.post(
    "/documents/{document_id}/process",
    response_model=DocumentResponse | ProcessingJobResponse,
    responses={
        200: {"model": DocumentResponse},
        202: {"model": ProcessingJobResponse},
    },
)
def process_document(
    document_id: UUID,
    user: CurrentUser,
    response: Response,
    db: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
    embedder: EmbeddingBackend = Depends(get_embedder),
    wait: bool = Query(
        True,
        description="If true (default), process on this request. If false, enqueue a durable job and return 202.",
    ),
) -> Document | ProcessingJobResponse:
    document = _authorized_document(db, _principal(user), document_id, "write")
    if not wait:
        try:
            if document.storage_path:
                storage.resolve(document.storage_path)
            job = enqueue_processing_job(db, document, _principal(user))
        except StoredFileMissingError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stored file not found",
            ) from exc
        except JobEnqueueError as exc:
            code = (
                status.HTTP_409_CONFLICT
                if exc.code == "conflict"
                else status.HTTP_404_NOT_FOUND
            )
            raise HTTPException(status_code=code, detail=exc.message) from exc
        response.status_code = status.HTTP_202_ACCEPTED
        return ProcessingJobResponse.model_validate(job)
    try:
        return process_stored_document(
            db,
            document,
            storage,
            embedder,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    except IngestionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except StoredFileMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored file not found",
        ) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Document contains no extractable text",
        ) from exc
    except ExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Unable to extract text from document",
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed",
        ) from exc


@router.get(
    "/documents/{document_id}/processing-jobs/{job_id}",
    response_model=ProcessingJobResponse,
)
def get_processing_job(
    document_id: UUID,
    job_id: UUID,
    user: CurrentUser,
    db: Session = Depends(get_db),
) -> ProcessingJobResponse:
    _authorized_document(db, _principal(user), document_id, "read")
    job = get_job(db, job_id)
    if job is None or job.document_id != document_id:
        raise _not_found()
    principal = _principal(user)
    if job.tenant_id != principal.tenant_id:
        raise _not_found()
    return ProcessingJobResponse.model_validate(job)


@router.get(
    "/documents/{document_id}/chunks",
    response_model=list[DocumentChunkResponse],
)
def list_document_chunks(
    document_id: UUID,
    user: CurrentUser,
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> list[DocumentChunk]:
    _authorized_document(db, _principal(user), document_id, "read")
    statement = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
        .offset(skip)
        .limit(limit)
    )
    return list(db.scalars(statement).all())


@router.get("/documents/{document_id}/download")
def download_document(
    document_id: UUID,
    user: CurrentUser,
    db: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
) -> FileResponse:
    document = _authorized_document(db, _principal(user), document_id, "read")
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


@router.get("/documents/{document_id}/permissions", response_model=list[PermissionGrantResponse])
def list_document_permissions(
    document_id: UUID,
    user: CurrentUser,
    db: Session = Depends(get_db),
) -> list[DocumentACL]:
    _authorized_document(db, _principal(user), document_id, "manage")
    return list_grants(db, document_id)


@router.put("/documents/{document_id}/permissions", response_model=PermissionGrantResponse)
def grant_document_permission(
    document_id: UUID,
    payload: PermissionGrantRequest,
    user: CurrentUser,
    db: Session = Depends(get_db),
) -> DocumentACL:
    principal = _principal(user)
    document = _authorized_document(db, principal, document_id, "manage")
    try:
        return grant_privilege(
            db,
            principal,
            document=document,
            target_user_id=payload.user_id,
            privilege=payload.privilege,
        )
    except PermissionError as exc:
        code = status.HTTP_403_FORBIDDEN if exc.code in {"forbidden", "escalation"} else status.HTTP_404_NOT_FOUND
        if exc.code == "invalid_privilege":
            code = status.HTTP_422_UNPROCESSABLE_CONTENT
        raise HTTPException(status_code=code, detail=exc.message) from exc


@router.delete("/documents/{document_id}/permissions/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_document_permission(
    document_id: UUID,
    user_id: UUID,
    user: CurrentUser,
    db: Session = Depends(get_db),
) -> None:
    principal = _principal(user)
    document = _authorized_document(db, principal, document_id, "manage")
    try:
        revoke_privilege(db, principal, document=document, target_user_id=user_id)
    except PermissionError as exc:
        code = status.HTTP_403_FORBIDDEN if exc.code == "forbidden" else status.HTTP_404_NOT_FOUND
        raise HTTPException(status_code=code, detail=exc.message) from exc


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(
    document_id: UUID,
    user: CurrentUser,
    db: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
) -> None:
    principal = _principal(user)
    document = _authorized_document(db, principal, document_id, "manage")
    storage_path = document.storage_path
    record_audit(
        db,
        action="document.delete",
        outcome="success",
        principal=principal,
        resource_type="document",
        resource_id=document.id,
    )
    db.delete(document)
    db.commit()
    storage.delete(storage_path)
