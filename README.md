# Enterprise AI Platform

Backend foundation for an Enterprise AI Knowledge & Workflow Automation Platform.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the API

From the repository root:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --app-dir backend
```

Open http://127.0.0.1:8000/health or http://127.0.0.1:8000/docs.

## PostgreSQL (local Docker)

Copy the example environment file, then start the database:

```bash
cp .env.example .env
docker compose up -d db
```

Stop it with:

```bash
docker compose stop db
```

Remove the container but keep data:

```bash
docker compose down
```

Check that Postgres is accepting connections:

```bash
docker compose ps
docker compose exec db pg_isready -U enterprise_user -d enterprise_ai
```

A healthy `db` service reports `status: healthy`. Data lives in the named volume `postgres_data`, so it survives `docker compose down`. Delete that volume only when you want a clean database:

```bash
docker compose down -v
```

Postgres is bound to `127.0.0.1:5432` (this machine only). With the API running, check connectivity at http://127.0.0.1:8000/db-health.

## Database migrations (Alembic)

Run Alembic from the project root with the virtualenv active and PostgreSQL running. Credentials come from `.env` via application settings; they are not stored in `alembic.ini`.

```bash
alembic current
alembic upgrade head
alembic downgrade -1
alembic revision --autogenerate -m "describe the change"
alembic upgrade head --sql
```

There are no application tables or revision files yet. `alembic current` should connect and report that no revision is applied. Use `--sql` for offline SQL generation without applying changes.

## Run tests

```bash
source .venv/bin/activate
pytest
```
