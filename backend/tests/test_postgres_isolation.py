import pytest
from sqlalchemy.engine import URL

from tests.postgres_isolation import require_isolated_url


def test_rejects_development_database_name() -> None:
    url = "postgresql+psycopg://u:p@127.0.0.1:5432/enterprise_ai"
    with pytest.raises(RuntimeError, match="development"):
        require_isolated_url(url, development_db="enterprise_ai")


def test_rejects_names_that_do_not_end_with_test() -> None:
    url = "postgresql+psycopg://u:p@127.0.0.1:5432/enterprise_ai_staging"
    with pytest.raises(RuntimeError, match="_test"):
        require_isolated_url(url)


def test_accepts_explicit_acl_test_database() -> None:
    url = "postgresql+psycopg://u:p@127.0.0.1:5432/enterprise_ai_acl_test"
    assert require_isolated_url(url, development_db="enterprise_ai") == url


def test_url_parser_does_not_require_printing_password() -> None:
    parsed = URL.create(
        "postgresql+psycopg",
        username="enterprise_user",
        password="secret-value",
        host="127.0.0.1",
        port=5432,
        database="enterprise_ai_acl_test",
    )
    assert parsed.database == "enterprise_ai_acl_test"
    assert "secret-value" not in parsed.render_as_string(hide_password=True)
