"""Development-only admin bootstrap. Never expose this as an HTTP endpoint.

Requires all of:
  BOOTSTRAP_ADMIN_ENABLED=true
  BOOTSTRAP_ADMIN_EMAIL
  BOOTSTRAP_ADMIN_PASSWORD (at least 10 characters)

Example:

  BOOTSTRAP_ADMIN_ENABLED=true \\
  BOOTSTRAP_ADMIN_EMAIL=admin@localhost \\
  BOOTSTRAP_ADMIN_PASSWORD='choose-a-long-local-password' \\
  PYTHONPATH=backend python -m app.bootstrap_admin
"""

from __future__ import annotations

import os
import sys

from app.core.database import SessionLocal
from app.core.security import is_valid_email, normalize_email
from app.services.users import upsert_admin


def main() -> int:
    enabled = os.getenv("BOOTSTRAP_ADMIN_ENABLED", "").strip().lower() in {"1", "true", "yes"}
    email = os.getenv("BOOTSTRAP_ADMIN_EMAIL", "")
    password = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")
    if not enabled:
        print("Refusing to bootstrap: set BOOTSTRAP_ADMIN_ENABLED=true in this shell only.", file=sys.stderr)
        return 1
    if not is_valid_email(email):
        print("Refusing to bootstrap: BOOTSTRAP_ADMIN_EMAIL is missing or invalid.", file=sys.stderr)
        return 1
    if len(password) < 10 or password != password.strip():
        print("Refusing to bootstrap: BOOTSTRAP_ADMIN_PASSWORD must be at least 10 characters.", file=sys.stderr)
        return 1
    db = SessionLocal()
    try:
        user = upsert_admin(db, email=email, password=password)
    finally:
        db.close()
    print(f"Admin ready for {normalize_email(email)} (id={user.id}). Password was not printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
