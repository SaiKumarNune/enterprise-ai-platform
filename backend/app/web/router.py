from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.api.agents import approve_agent, inspect_agent, reject_agent, run_agent
from app.api.auth import _issue_token
from app.api.documents import (
    get_document,
    get_processing_job,
    list_documents,
    process_document,
    upload_document,
)
from app.api.rag import ask_rag
from app.api.search import search_documents
from app.core.database import get_db
from app.models.user import APPROVER_ROLES, User
from app.schemas.agent import AgentRunRequest
from app.schemas.rag import RagAskRequest
from app.schemas.search import SearchRequest
from app.services.embeddings import EmbeddingBackend, get_embedder
from app.services.llm import LLMBackend, get_llm
from app.services.storage import LocalStorage, get_storage
from app.web.deps import error_detail, get_ui_user, is_hx, page_context, templates

router = APIRouter()
UiUser = Annotated[User, Depends(get_ui_user)]

MAX_RECENT_WORKFLOWS = 20


def _render(name: str, request: Request, user: User | None = None, status_code: int = 200, **extra):
    return templates.TemplateResponse(
        request,
        name,
        page_context(request, user, **extra),
        status_code=status_code,
    )


def _remember_workflow(request: Request, workflow_id: UUID) -> None:
    items = [str(workflow_id)]
    for existing in request.session.get("workflow_ids", []):
        if existing != str(workflow_id):
            items.append(existing)
    request.session["workflow_ids"] = items[:MAX_RECENT_WORKFLOWS]


def _parse_uuid(raw: str | None) -> UUID | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        return UUID(str(raw).strip())
    except ValueError:
        return None


@router.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app/", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/app/login")
def login_page(request: Request):
    if request.session.get("access_token"):
        return RedirectResponse(url="/app/", status_code=status.HTTP_303_SEE_OTHER)
    return _render("login.html", request, error=None, email="")


@router.post("/app/login")
def login_submit(
    request: Request,
    db: Session = Depends(get_db),
    email: str = Form(""),
    password: str = Form(""),
):
    try:
        token = _issue_token(db, email, password)
    except HTTPException as exc:
        return _render(
            "login.html",
            request,
            error=error_detail(exc),
            email=email.strip(),
            status_code=exc.status_code if exc.status_code in {401, 429, 503} else 401,
        )
    request.session["access_token"] = token.access_token
    request.session["workflow_ids"] = []
    return RedirectResponse(url="/app/", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/app/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/app/login", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/app/")
def dashboard(request: Request, user: UiUser):
    return _render("dashboard.html", request, user)


@router.get("/app/documents")
def documents_page(request: Request, user: UiUser, db: Session = Depends(get_db)):
    documents = list_documents(user, db, skip=0, limit=20)
    return _render("documents.html", request, user, documents=documents, job=None, error=None)


@router.post("/app/documents/upload")
def documents_upload(
    request: Request,
    user: UiUser,
    db: Session = Depends(get_db),
    storage: LocalStorage = Depends(get_storage),
    embedder: EmbeddingBackend = Depends(get_embedder),
    file: UploadFile = File(...),
):
    error = None
    job = None
    try:
        document = upload_document(user=user, file=file, db=db, storage=storage)
        dummy = Response()
        job = process_document(
            document.id,
            user,
            dummy,
            db,
            storage,
            embedder,
            wait=False,
        )
    except HTTPException as exc:
        error = error_detail(exc)
    documents = list_documents(user, db, skip=0, limit=20)
    template = "partials/documents_panel.html" if is_hx(request) else "documents.html"
    return _render(template, request, user, documents=documents, job=job, error=error)


@router.get("/app/documents/{document_id}/jobs/{job_id}")
def documents_job(
    document_id: UUID,
    job_id: UUID,
    request: Request,
    user: UiUser,
    db: Session = Depends(get_db),
):
    error = None
    job = None
    try:
        job = get_processing_job(document_id, job_id, user, db)
    except HTTPException as exc:
        error = error_detail(exc)
    return _render("partials/job_status.html", request, user, job=job, error=error)


@router.get("/app/search")
def search_page(request: Request, user: UiUser):
    return _render(
        "search.html",
        request,
        user,
        hits=None,
        error=None,
        query="",
        retrieval_mode="hybrid",
    )


@router.post("/app/search")
def search_submit(
    request: Request,
    user: UiUser,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
    query: str = Form(""),
    retrieval_mode: str = Form("hybrid"),
):
    hits = None
    error = None
    mode = retrieval_mode if retrieval_mode in {"lexical", "semantic", "hybrid"} else "hybrid"
    if not query.strip():
        error = None
        hits = []
        empty_query = True
    else:
        empty_query = False
        try:
            payload = SearchRequest(query=query, retrieval_mode=mode, include_diagnostics=True)
            hits = search_documents(payload, user, db, embedder)
        except HTTPException as exc:
            error = error_detail(exc)
            hits = []
    template = "partials/search_results.html" if is_hx(request) else "search.html"
    return _render(
        template,
        request,
        user,
        hits=hits,
        error=error,
        query=query,
        retrieval_mode=mode,
        empty_query=empty_query,
    )


@router.get("/app/ask")
def rag_page(request: Request, user: UiUser):
    return _render("rag.html", request, user, result=None, error=None, question="")


@router.post("/app/ask")
def rag_submit(
    request: Request,
    user: UiUser,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
    llm: LLMBackend = Depends(get_llm),
    question: str = Form(""),
):
    result = None
    error = None
    if not question.strip():
        error = "Enter a question."
    else:
        try:
            payload = RagAskRequest(question=question)
            result = ask_rag(payload, user, db, embedder, llm)
        except HTTPException as exc:
            error = error_detail(exc)
    template = "partials/rag_result.html" if is_hx(request) else "rag.html"
    return _render(template, request, user, result=result, error=error, question=question)


@router.get("/app/agents")
def agents_page(request: Request, user: UiUser, db: Session = Depends(get_db)):
    workflow = None
    error = None
    recent = request.session.get("workflow_ids") or []
    if recent:
        try:
            workflow = inspect_agent(UUID(recent[0]), user, db)
        except HTTPException:
            workflow = None
    return _render(
        "agents.html",
        request,
        user,
        workflow=workflow,
        error=error,
        request_text="",
        recent=recent,
    )


@router.post("/app/agents/run")
def agents_run(
    request: Request,
    user: UiUser,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
    llm: LLMBackend = Depends(get_llm),
    request_text: str = Form(""),
):
    workflow = None
    error = None
    if not request_text.strip():
        error = "Enter a request for the agent."
    else:
        try:
            payload = AgentRunRequest(request=request_text)
            workflow = run_agent(payload, user, db, embedder, llm)
            _remember_workflow(request, workflow.workflow_id)
        except HTTPException as exc:
            error = error_detail(exc)
    template = "partials/workflow_card.html" if is_hx(request) else "agents.html"
    return _render(
        template,
        request,
        user,
        workflow=workflow,
        error=error,
        request_text=request_text,
        recent=request.session.get("workflow_ids") or [],
    )


@router.post("/app/agents/lookup")
def agents_lookup(
    request: Request,
    user: UiUser,
    db: Session = Depends(get_db),
    workflow_id: str = Form(""),
):
    workflow = None
    error = None
    parsed = _parse_uuid(workflow_id)
    if parsed is None:
        error = "Enter a valid workflow id."
    else:
        try:
            workflow = inspect_agent(parsed, user, db)
            _remember_workflow(request, parsed)
        except HTTPException as exc:
            error = error_detail(exc)
    template = "partials/workflow_card.html" if is_hx(request) else "agents.html"
    return _render(
        template,
        request,
        user,
        workflow=workflow,
        error=error,
        request_text="",
        recent=request.session.get("workflow_ids") or [],
    )


@router.post("/app/agents/{workflow_id}/approve")
def agents_approve(
    workflow_id: UUID,
    request: Request,
    user: UiUser,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
    llm: LLMBackend = Depends(get_llm),
):
    return _decide(request, user, db, embedder, llm, workflow_id, approve=True)


@router.post("/app/agents/{workflow_id}/reject")
def agents_reject(
    workflow_id: UUID,
    request: Request,
    user: UiUser,
    db: Session = Depends(get_db),
    embedder: EmbeddingBackend = Depends(get_embedder),
    llm: LLMBackend = Depends(get_llm),
):
    return _decide(request, user, db, embedder, llm, workflow_id, approve=False)


def _decide(
    request: Request,
    user: User,
    db: Session,
    embedder: EmbeddingBackend,
    llm: LLMBackend,
    workflow_id: UUID,
    *,
    approve: bool,
):
    workflow = None
    error = None
    status_code = 200
    if user.role not in APPROVER_ROLES:
        error = "Insufficient permissions"
        status_code = status.HTTP_403_FORBIDDEN
    else:
        try:
            if approve:
                workflow = approve_agent(workflow_id, user, None, db, embedder, llm)
            else:
                workflow = reject_agent(workflow_id, user, None, db, embedder, llm)
            _remember_workflow(request, workflow_id)
        except HTTPException as exc:
            error = error_detail(exc)
            status_code = exc.status_code
    template = "partials/workflow_card.html" if is_hx(request) else "agents.html"
    return _render(
        template,
        request,
        user,
        workflow=workflow,
        error=error,
        request_text="",
        recent=request.session.get("workflow_ids") or [],
        status_code=status_code,
    )
