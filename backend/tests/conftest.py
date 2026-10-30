import pytest

from app.main import app
from app.services.embeddings import get_embedder
from tests.fake_embeddings import HashEmbedder


@pytest.fixture(autouse=True)
def use_fake_embedder() -> None:
    embedder = HashEmbedder()
    app.dependency_overrides[get_embedder] = lambda: embedder
    yield
    app.dependency_overrides.pop(get_embedder, None)
