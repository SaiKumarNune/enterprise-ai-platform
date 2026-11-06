from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa


def generate_rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeOidcProvider:
    """In-process authorization-code stand-in. No network and no real IdP."""

    def __init__(
        self,
        private_key,
        *,
        issuer: str = "https://idp.test",
        client_id: str = "enterprise-ai",
        redirect_uri: str = "http://127.0.0.1:8000/auth/oidc/callback",
    ) -> None:
        self.private_key = private_key
        self.public_key = private_key.public_key()
        self.issuer = issuer
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.authorization_endpoint = f"{issuer}/authorize"
        self.codes: dict[str, dict] = {}

    def authorization_url(self, *, state: str, nonce: str, code_challenge: str) -> str:
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self.client_id,
                "redirect_uri": self.redirect_uri,
                "scope": "openid email",
                "state": state,
                "nonce": nonce,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }
        )
        return f"{self.authorization_endpoint}?{query}"

    def issue_code(self, *, nonce: str, email: str, sub: str, **overrides) -> str:
        code = secrets.token_urlsafe(16)
        self.codes[code] = {"nonce": nonce, "email": email, "sub": sub, **overrides}
        return code

    def encode_id_token(
        self,
        *,
        nonce: str,
        email: str,
        sub: str,
        exp_delta: int = 3600,
        audience: str | None = None,
        issuer: str | None = None,
        extra: dict | None = None,
        key=None,
        algorithm: str = "RS256",
    ) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "iss": issuer if issuer is not None else self.issuer,
            "aud": self.client_id if audience is None else audience,
            "sub": sub,
            "email": email,
            "email_verified": True,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=exp_delta)).timestamp()),
            "nonce": nonce,
        }
        if extra:
            payload.update(extra)
        return jwt.encode(payload, key or self.private_key, algorithm=algorithm)

    def fetch_id_token(self, *, code: str, code_verifier: str) -> str:
        try:
            record = self.codes.pop(code)
        except KeyError as exc:
            raise ValueError("unknown authorization code") from exc
        return self.encode_id_token(
            nonce=record["nonce"],
            email=record["email"],
            sub=record["sub"],
            exp_delta=record.get("exp_delta", 3600),
            audience=record.get("audience"),
            issuer=record.get("issuer"),
            extra=record.get("extra"),
            key=record.get("key"),
            algorithm=record.get("algorithm", "RS256"),
        )

    def signing_key(self, id_token: str):
        return self.public_key
