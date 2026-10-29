# Enterprise AI Platform

Backend foundation for an Enterprise AI Knowledge & Workflow Automation Platform.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

## Run the API

From the repository root, with PostgreSQL running:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --app-dir backend
```

Open http://127.0.0.1:8000/health or http://127.0.0.1:8000/docs.

## PostgreSQL (local Docker)

```bash
docker compose up -d db
docker compose exec db pg_isready -U enterprise_user -d enterprise_ai
docker compose down        # keep volume
docker compose down -v     # wipe database data
```

Postgres is bound to `127.0.0.1:5432`.

## Database migrations (Alembic)

```bash
alembic current
alembic upgrade head
alembic revision --autogenerate -m "describe the change"
```

## Document API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/documents` | Metadata-only create |
| `GET` | `/documents` | List; `skip` default 0, `limit` default 20, max 100 |
| `GET` | `/documents/{document_id}` | Fetch one document |
| `PATCH` | `/documents/{document_id}` | `filename` and `content_type` only |
| `DELETE` | `/documents/{document_id}` | Delete metadata, chunks (cascade), and best-effort file cleanup |
| `POST` | `/documents/upload` | Multipart PDF or UTF-8 text upload |
| `GET` | `/documents/{document_id}/download` | Download stored file |
| `POST` | `/documents/{document_id}/process` | Extract text, chunk, persist chunks |
| `GET` | `/documents/{document_id}/chunks` | List chunks; `skip`/`limit`, ordered by `chunk_index` |

Uploads: PDF and UTF-8 text only, 10 MB max. Files live in `UPLOAD_DIR` (default `uploads/`, gitignored) under a server-generated UUID name.

### Ingestion

`POST /documents/{id}/process` is **synchronous**. It reads the stored file, extracts text with pypdf (PDF) or UTF-8 decode (TXT), splits into overlapping chunks (default 1000 characters, 200 overlap; `CHUNK_SIZE` / `CHUNK_OVERLAP`), and replaces any previous chunks in one database transaction.

Status flow:

- `uploaded` or `failed` or `ready` → `processing` → `ready` (success) or `failed` (error)
- `processing` cannot be started again (`409`) until it finishes
- metadata-only documents (`storage_path` empty) return `409`
- missing stored files return `404`
- empty/unreadable files return `422` and set `failed` without deleting existing chunks

Reprocessing a `ready` document deletes old chunks and inserts the new set in the same commit. If that commit fails, the previous chunks remain.

`PATCH` cannot change `status`. Processing state is owned by `/process` so clients cannot mark a document `ready` without chunks or `processing` without a job.

Scanned/image-only PDFs have no extractable text and need OCR, which is not implemented in this phase.

### Examples

```bash
curl -X POST http://127.0.0.1:8000/documents/upload -F "file=@./handbook.pdf"
curl -X POST http://127.0.0.1:8000/documents/<document-id>/process
curl "http://127.0.0.1:8000/documents/<document-id>/chunks?skip=0&limit=20"
curl -O -J http://127.0.0.1:8000/documents/<document-id>/download
```

## Run tests

```bash
source .venv/bin/activate
pytest
```

Tests use an in-memory fake session and temporary directories. They do not write to the development PostgreSQL database.
