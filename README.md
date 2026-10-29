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
docker compose ps
docker compose exec db pg_isready -U enterprise_user -d enterprise_ai
docker compose stop db
docker compose down        # keep volume
docker compose down -v     # wipe database data
```

Postgres is bound to `127.0.0.1:5432`. Check connectivity at http://127.0.0.1:8000/db-health.

## Database migrations (Alembic)

Run from the project root with the virtualenv active and PostgreSQL running:

```bash
alembic current
alembic upgrade head
alembic downgrade -1
alembic revision --autogenerate -m "describe the change"
```

## Document API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/documents` | Metadata-only create |
| `GET` | `/documents` | List; `skip` default 0, `limit` default 20, max 100. Ordered by `created_at DESC`, `id DESC` |
| `GET` | `/documents/{document_id}` | Fetch one document |
| `PATCH` | `/documents/{document_id}` | Partial update of `filename`, `content_type`, `status` |
| `DELETE` | `/documents/{document_id}` | Delete metadata and best-effort file cleanup |
| `POST` | `/documents/upload` | Multipart file upload (`file` field) |
| `GET` | `/documents/{document_id}/download` | Download stored file |

Allowed upload types: PDF and UTF-8 plain text. Maximum size: 10 MB (`MAX_UPLOAD_BYTES`). Type is detected from file content, not the client filename or MIME type.

Files are stored under `UPLOAD_DIR` (default `uploads/` at the project root, gitignored) using a server-generated UUID filename. The original filename is saved only as metadata.

### Upload and download examples

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@./handbook.pdf"

curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@./notes.txt"

curl -O -J http://127.0.0.1:8000/documents/<document-id>/download
```

`status` may be `uploaded`, `processing`, `ready`, or `failed`.

## Run tests

```bash
source .venv/bin/activate
pytest
```

Tests use an in-memory fake session and temporary directories. They do not write to the development PostgreSQL database.
