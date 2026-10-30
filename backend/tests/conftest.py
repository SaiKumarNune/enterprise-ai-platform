import os

os.environ["JWT_SECRET"] = "pytest-only-jwt-secret-key-not-for-prod"

import pytest

from app.main import app
from app.services.embeddings import get_embedder
from app.services.llm import get_llm
from tests.fake_embeddings import HashEmbedder
from tests.fake_llm import ScriptedLLM


@pytest.fixture(autouse=True)
def use_fake_embedder() -> None:
    embedder = HashEmbedder()
    app.dependency_overrides[get_embedder] = lambda: embedder
    yield
    app.dependency_overrides.pop(get_embedder, None)


@pytest.fixture(autouse=True)
def use_fake_llm() -> None:
    llm = ScriptedLLM(
        "I don't have enough information in the retrieved documents to answer that question."
    )
    app.dependency_overrides[get_llm] = lambda: llm
    yield
    app.dependency_overrides.pop(get_llm, None)
