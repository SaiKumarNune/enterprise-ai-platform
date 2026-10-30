from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document
from app.schemas.rag import RagAskRequest, RagAskResponse, RagSourceResponse
from app.services.embeddings import EmbeddingBackend, EmbeddingError, get_embedder
from app.services.llm import (
    LLMBackend,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    get_llm,
)
from app.services.rag import answer_question

router = APIRouter()


@router.post("/rag/ask", response_model=RagAskResponse)
def ask_rag(
    payload: RagAskRequest,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
    llm: LLMBackend = Depends(get_llm),
) -> RagAskResponse:
    if payload.document_id is not None:
        document = db.get(Document, payload.document_id)
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )
    try:
        result = answer_question(
            db,
            embedder,
            llm,
            question=payload.question,
            top_k=payload.top_k,
            document_id=payload.document_id,
        )
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to embed question",
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The local language model timed out",
        ) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local language model is unavailable",
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The local language model failed",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Question answering failed",
        ) from exc

    return RagAskResponse(
        answer=result.answer,
        sources=[
            RagSourceResponse(
                source_id=source.source_id,
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                chunk_index=source.chunk_index,
                page_number=source.page_number,
                similarity=source.similarity,
            )
            for source in result.sources
        ],
        retrieval_count=result.retrieval_count,
        grounded=result.grounded,
        model=result.model,
    )
