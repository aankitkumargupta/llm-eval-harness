"""
Sign-in gate for the browser UI: a landing page, a shared pilot password, a
name and function for attribution, and a role that is enforced server-side.

What it is: a single gate in front of a loopback-only tool that can spend a
provider key. The password comes from the environment
(HARNESS_PILOT_PASSWORD) and is hashed with PBKDF2-HMAC-SHA256 once at start;
an attempt is hashed the same way and compared in constant time, so the
plain value is never held for comparison. Sessions are random tokens kept
in this process, carried in an HttpOnly cookie, and expire after twelve
hours. The Assurance Lead role is required for the two endpoints that start
paid runs; an Evaluator reads everything else.

What it is not: user accounts, SSO or MFA. The default password is the
pilot's shared one and is announced on the console at start so nobody
mistakes it for a secret. Production would put real authentication in
front of this server; the gate is not a reason to expose it beyond
loopback.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from collections.abc import Callable

from .api import ApiError

PASSWORD_ENV = "HARNESS_PILOT_PASSWORD"
DEFAULT_PASSWORD = "Maha@2027"          # built-in pilot default; the console says so when it is in use
SESSION_COOKIE = "harness_session"
SESSION_TTL_S = 12 * 3600
PBKDF2_ROUNDS = 200_000
ROLES = ("Assurance Lead", "Evaluator")
SPENDING_ROUTES = ("/api/run", "/api/run-profile")
OPEN_ROUTES = ("/api/auth/login", "/api/auth/logout", "/api/auth/me")
MAX_FIELD = 120


class AuthError(ApiError):
    """A sign-in problem, rendered with its HTTP status."""


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)


class Sessions:
    """Password check and the live session table for one server process."""

    def __init__(self, password: str | None = None, *, ttl_s: int = SESSION_TTL_S,
                 clock: Callable[[], float] = time.time) -> None:
        configured = password if password is not None else os.environ.get(PASSWORD_ENV, "")
        self.using_default = not configured
        self._salt = os.urandom(16)
        self._hash = _hash(configured or DEFAULT_PASSWORD, self._salt)
        self._ttl = ttl_s
        self._clock = clock
        self._live: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    def login(self, name: str, department: str, role: str, password: str) -> tuple[str, dict]:
        """Check the attempt; return (token, user) or raise AuthError."""
        attempt = _hash(str(password or ""), self._salt)
        if not hmac.compare_digest(attempt, self._hash):
            raise AuthError("Incorrect pilot password.", 401)
        name = str(name or "").strip()[:MAX_FIELD]
        if not name:
            raise AuthError("Name is required.", 400)
        if role not in ROLES:
            raise AuthError(f"Role must be one of: {', '.join(ROLES)}.", 400)
        user = {"name": name, "department": str(department or "").strip()[:MAX_FIELD],
                "role": role, "signed_in_at": self._clock()}
        token = secrets.token_urlsafe(32)
        self._live[token] = {**user, "expires": self._clock() + self._ttl}
        return token, user

    def user(self, token: str | None) -> dict | None:
        """The user behind a token, or None when unknown or expired."""
        if not token:
            return None
        rec = self._live.get(token)
        if rec is None:
            return None
        if rec["expires"] <= self._clock():
            self._live.pop(token, None)
            return None
        return {k: v for k, v in rec.items() if k != "expires"}

    def logout(self, token: str | None) -> None:
        if token:
            self._live.pop(token, None)

    # ------------------------------------------------------------------ #
    @staticmethod
    def token_from_cookie(header: str | None) -> str | None:
        for part in (header or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == SESSION_COOKIE and v:
                return v
        return None

    @staticmethod
    def cookie_for(token: str) -> str:
        return (f"{SESSION_COOKIE}={token}; HttpOnly; SameSite=Lax; Path=/; "
                f"Max-Age={SESSION_TTL_S}")

    @staticmethod
    def cleared_cookie() -> str:
        return f"{SESSION_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0"


def requires_role(route: str) -> str | None:
    """The role a route needs beyond being signed in, if any."""
    return "Assurance Lead" if route in SPENDING_ROUTES else None
