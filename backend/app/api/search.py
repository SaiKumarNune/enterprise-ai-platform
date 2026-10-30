from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document
from app.schemas.search import SearchHit, SearchRequest
from app.services.embeddings import EmbeddingBackend, EmbeddingError, get_embedder
from app.services.search import search_similar_chunks

router = APIRouter()


@router.post("/search", response_model=list[SearchHit])
def search_documents(
    payload: SearchRequest,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
) -> list[SearchHit]:
    if payload.document_id is not None:
        document = db.get(Document, payload.document_id)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
    try:
        query_embedding = embedder.embed_texts([payload.query])[0]
        rows = search_similar_chunks(
            db,
            query_embedding,
            top_k=payload.top_k,
            document_id=payload.document_id,
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
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            page_number=chunk.page_number,
            similarity=round(score, 6),
        )
        for chunk, score in rows
    ]
