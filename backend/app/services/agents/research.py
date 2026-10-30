from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.services.agent_tools import dispatch_tool
from app.services.authorization import Principal
from app.services.embeddings import EmbeddingBackend
from app.services.llm import LLMBackend
from app.services.agents.validator import ground_informational_answer
from app.services.rag import (
    INSUFFICIENT_ANSWER,
    SYSTEM_PROMPT,
    build_user_prompt,
    inspect_answer,
    source_dicts_for_citations,
)


def run_research(
    db: Session,
    embedder: EmbeddingBackend,
    *,
    question: str,
    top_k: int,
    document_id: str | None,
    principal: Principal,
) -> dict[str, Any]:
    parsed_document_id = UUID(document_id) if document_id else None
    result = dispatch_tool(
        "knowledge_search",
        db=db,
        embedder=embedder,
        question=question,
        top_k=top_k,
        document_id=parsed_document_id,
        principal=principal,
    )
    return {
        "excerpts": result["excerpts"],
        "sources": result["sources"],
        "retrieval_count": result["retrieval_count"],
        "selected_count": len(result["excerpts"]),
    }


def synthesize_informational_answer(
    llm: LLMBackend,
    *,
    question: str,
    excerpts: list[dict[str, Any]],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    valid_ids = {item["source_id"] for item in sources}
    selected = selected_from_excerpts(excerpts, sources)
    raw_answer = llm.generate(SYSTEM_PROMPT, build_user_prompt(question, selected))
    check = inspect_answer(raw_answer, valid_ids)
    answer = check.cleaned_answer if not check.abstained else INSUFFICIENT_ANSWER
    abstained = check.abstained
    grounded = check.grounded
    citation_valid = check.citation_valid
    grounding_issues: list[dict[str, str]] = []
    if not abstained:
        grounded_answer = ground_informational_answer(answer, excerpts)
        answer = grounded_answer["answer"]
        grounding_issues = grounded_answer["issues"]
        if grounded_answer["abstain"]:
            abstained = True
            grounded = False
            citation_valid = True
    return {
        "answer": answer,
        "sources": [] if abstained else source_dicts_for_citations(sources, check.citation_ids),
        "draft": None,
        "grounded": grounded,
        "citation_valid": citation_valid,
        "abstained": abstained,
        "status": "completed",
        "grounding_issues": grounding_issues,
    }


def selected_from_excerpts(excerpts: list[dict[str, Any]], sources: list[dict[str, Any]]):
    by_id = {item["source_id"]: item for item in sources}

    class _Chunk:
        def __init__(self, excerpt: dict[str, Any], meta: dict[str, Any] | None) -> None:
            self.content = excerpt["content"]
            self.document_id = (meta or {}).get("document_id", "")
            self.id = (meta or {}).get("chunk_id", "")
            self.page_number = (meta or {}).get("page_number")

    return [
        (_Chunk(excerpt, by_id.get(excerpt["source_id"])), 1.0)
        for excerpt in excerpts
    ]
