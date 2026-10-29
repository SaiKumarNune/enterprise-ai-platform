# Enterprise AI Platform

Backend foundation for an Enterprise AI Knowledge & Workflow Automation Platform.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install fastapi "uvicorn[standard]" pytest httpx
```

## Run the API

From the repository root:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --app-dir backend
```

Open http://127.0.0.1:8000/health or http://127.0.0.1:8000/docs.

## Run tests

```bash
source .venv/bin/activate
pytest
```
