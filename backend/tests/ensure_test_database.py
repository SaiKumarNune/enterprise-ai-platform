"""Create enterprise_ai_acl_test if missing. Never drops enterprise_ai."""

from __future__ import annotations

from tests.postgres_isolation import ensure_isolated_database


def main() -> None:
    ensure_isolated_database("enterprise_ai_acl_test")
    print("Isolated PostgreSQL database ready: enterprise_ai_acl_test")


if __name__ == "__main__":
    main()
