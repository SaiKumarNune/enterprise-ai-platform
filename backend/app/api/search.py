from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.search import SearchHit, SearchRequest
from app.services.authorization import DocumentAccessDenied, Principal
from app.services.embeddings import EmbeddingBackend, EmbeddingError, get_embedder
from app.services.retrieval import HYBRID, LEXICAL, SEMANTIC, normalize_retrieval_mode, retrieve_chunks

router = APIRouter()
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post("/search", response_model=list[SearchHit])
def search_documents(
    payload: SearchRequest,
    user: CurrentUser,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
) -> list[SearchHit]:
    principal = Principal.from_user(user)
    try:
        mode = normalize_retrieval_mode(payload.retrieval_mode)
        query_embedding = None
        if mode in {SEMANTIC, HYBRID}:
            try:
                query_embedding = embedder.embed_texts([payload.query])[0]
            except Exception:
                if mode == SEMANTIC:
                    raise
                mode = LEXICAL
                query_embedding = None
        hits = retrieve_chunks(
            db,
            query=payload.query,
            principal=principal,
            top_k=payload.top_k,
            mode=mode,
            query_embedding=query_embedding,
            document_id=payload.document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    except DocumentAccessDenied:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to embed search query",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed",
        ) from exc

    return [
        SearchHit(
            chunk_id=hit.chunk.id,
            document_id=hit.chunk.document_id,
            chunk_index=hit.chunk.chunk_index,
            content=hit.chunk.content,
            page_number=hit.chunk.page_number,
            similarity=round(hit.score, 6),
            retrieval_mode=hit.mode if payload.include_diagnostics else None,
        )
        for hit in hits
    ]
