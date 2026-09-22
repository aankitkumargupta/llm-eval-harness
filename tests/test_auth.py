"""
The sign-in gate in front of the browser UI (harness/web/auth.py and the
server's use of it). Offline: the handler is driven with an in-memory
request, no socket. The password is hashed before comparison, sessions
expire, the app and its API are unreachable without a session, the two
spending endpoints need the Assurance Lead role, and the landing page is
what an anonymous visitor gets.
"""

from __future__ import annotations

import io
import json
from email.message import Message

import pytest

from harness.web import auth as A
from harness.web import server


# --------------------------------------------------------------------------- #
#  Sessions
# --------------------------------------------------------------------------- #
def test_wrong_password_is_rejected_and_the_default_is_flagged(monkeypatch):
    monkeypatch.delenv(A.PASSWORD_ENV, raising=False)
    s = A.Sessions()
    assert s.using_default is True
    with pytest.raises(A.AuthError) as e:
        s.login("A", "Risk", "Evaluator", "nope")
    assert e.value.status == 401
    token, user = s.login("A", "Risk", "Evaluator", A.DEFAULT_PASSWORD)
    assert user == {"name": "A", "department": "Risk", "role": "Evaluator", "signed_in_at": user["signed_in_at"]}
    assert s.user(token)["name"] == "A"


def test_password_comes_from_the_environment_and_is_not_stored_plain(monkeypatch):
    monkeypatch.setenv(A.PASSWORD_ENV, "correct horse")
    s = A.Sessions()
    assert s.using_default is False
    with pytest.raises(A.AuthError):
        s.login("A", "Risk", "Evaluator", A.DEFAULT_PASSWORD)
    token, _ = s.login("A", "Risk", "Evaluator", "correct horse")
    assert token
    assert "correct horse" not in repr(vars(s)), "the plain password must not be held"


def test_name_and_role_are_required_and_fields_are_capped():
    s = A.Sessions(password="p")
    with pytest.raises(A.AuthError, match="Name is required"):
        s.login("  ", "Risk", "Evaluator", "p")
    with pytest.raises(A.AuthError, match="Role must be"):
        s.login("A", "Risk", "Admin", "p")
    _, user = s.login("x" * 500, "y" * 500, "Assurance Lead", "p")
    assert len(user["name"]) == A.MAX_FIELD and len(user["department"]) == A.MAX_FIELD


def test_sessions_expire_and_logout_forgets():
    now = [1000.0]
    s = A.Sessions(password="p", ttl_s=60, clock=lambda: now[0])
    token, _ = s.login("A", "Risk", "Evaluator", "p")
    assert s.user(token)
    now[0] += 61
    assert s.user(token) is None
    token2, _ = s.login("A", "Risk", "Evaluator", "p")
    s.logout(token2)
    assert s.user(token2) is None
    assert s.user(None) is None and s.user("unknown") is None


def test_cookie_helpers():
    assert A.Sessions.token_from_cookie("a=1; harness_session=tok; b=2") == "tok"
    assert A.Sessions.token_from_cookie(None) is None
    assert "HttpOnly" in A.Sessions.cookie_for("tok") and "SameSite=Lax" in A.Sessions.cookie_for("tok")
    assert "Max-Age=0" in A.Sessions.cleared_cookie()
    assert A.requires_role("/api/run") == "Assurance Lead"
    assert A.requires_role("/api/run-profile") == "Assurance Lead"
    assert A.requires_role("/api/profiles") is None


# --------------------------------------------------------------------------- #
#  The handler, driven without a socket
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, raw: bytes):
        head, _, body = raw.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        self.status = int(lines[0].split(" ")[1])
        self.headers = {}
        for line in lines[1:]:
            k, _, v = line.partition(": ")
            self.headers.setdefault(k.lower(), []).append(v)
        self.body = body

    def json(self):
        return json.loads(self.body.decode("utf-8"))

    def cookie_token(self):
        for c in self.headers.get("set-cookie", []):
            tok = A.Sessions.token_from_cookie(c.split(";")[0])
            if tok:
                return tok
        return None


def _call(handler_cls, method: str, path: str, *, body: dict | None = None, cookie: str | None = None) -> _Resp:
    h = handler_cls.__new__(handler_cls)
    h.command = method
    h.path = path
    h.request_version = "HTTP/1.1"
    h.requestline = f"{method} {path} HTTP/1.1"
    h.client_address = ("127.0.0.1", 0)
    h.close_connection = True
    h.headers = Message()
    payload = json.dumps(body).encode("utf-8") if body is not None else b""
    h.headers["Content-Length"] = str(len(payload))
    if cookie:
        h.headers["Cookie"] = f"{A.SESSION_COOKIE}={cookie}"
    h.rfile = io.BytesIO(payload)
    h.wfile = io.BytesIO()
    getattr(h, f"do_{method}")()
    return _Resp(h.wfile.getvalue())


@pytest.fixture
def handler(tmp_path):
    srv_cls = server.make_handler(store_path=str(tmp_path / "traces"), models_cfg={}, run_cfg={},
                                  budget=0.0, quiet=True, sessions=A.Sessions(password="pilot"))
    return srv_cls


def test_anonymous_visitor_gets_the_landing_and_no_api(handler):
    home = _call(handler, "GET", "/")
    assert home.status == 200
    assert b"Secure Sign-In" in home.body and b'id="main"' not in home.body
    assert _call(handler, "GET", "/api/profiles").status == 401
    assert _call(handler, "GET", "/api/case-studies").json()["error"].startswith("Sign in")
    # Static assets and the sign-in endpoints are reachable without a session.
    assert _call(handler, "GET", "/landing.css").status == 200
    assert _call(handler, "GET", "/api/auth/me").json() == {"user": None}


def test_sign_in_sets_a_session_cookie_and_opens_the_app(handler):
    bad = _call(handler, "POST", "/api/auth/login",
                body={"name": "A", "department": "Risk", "role": "Evaluator", "password": "wrong"})
    assert bad.status == 401 and bad.cookie_token() is None
    ok = _call(handler, "POST", "/api/auth/login",
               body={"name": "A", "department": "Risk", "role": "Evaluator", "password": "pilot"})
    assert ok.status == 200
    tok = ok.cookie_token()
    assert tok and "HttpOnly" in ok.headers["set-cookie"][0]
    home = _call(handler, "GET", "/", cookie=tok)
    assert b'id="main"' in home.body, "the app, not the landing, once signed in"
    me = _call(handler, "GET", "/api/auth/me", cookie=tok).json()
    assert me["user"]["name"] == "A" and me["user"]["role"] == "Evaluator"
    assert _call(handler, "GET", "/api/benchmarks", cookie=tok).status == 200


def test_evaluators_cannot_start_paid_runs_but_leads_can_reach_the_endpoint(handler):
    ev = _call(handler, "POST", "/api/auth/login",
               body={"name": "E", "department": "Risk", "role": "Evaluator", "password": "pilot"}).cookie_token()
    denied = _call(handler, "POST", "/api/run-profile", body={"profile": "x", "models": []}, cookie=ev)
    assert denied.status == 403 and "Assurance Lead" in denied.json()["error"]
    lead = _call(handler, "POST", "/api/auth/login",
                 body={"name": "L", "department": "Risk", "role": "Assurance Lead", "password": "pilot"}).cookie_token()
    reached = _call(handler, "POST", "/api/run-profile", body={"profile": "", "models": []}, cookie=lead)
    assert reached.status != 403 and reached.status != 401


def test_logout_clears_the_session(handler):
    tok = _call(handler, "POST", "/api/auth/login",
                body={"name": "A", "department": "Risk", "role": "Evaluator", "password": "pilot"}).cookie_token()
    out = _call(handler, "POST", "/api/auth/logout", cookie=tok)
    assert out.status == 200 and "Max-Age=0" in out.headers["set-cookie"][0]
    assert _call(handler, "GET", "/api/profiles", cookie=tok).status == 401


def test_the_app_shell_shows_the_user_and_a_sign_out():
    js = (server.STATIC / "app.js").read_text(encoding="utf-8")
    html = (server.STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="userchip"' in html and 'id="logoutbtn"' in html
    assert "/api/auth/me" in js and "/api/auth/logout" in js
    landing = (server.STATIC / "landing.html").read_text(encoding="utf-8")
    for needle in ("Assurance Lead", "Evaluator", "Shared pilot password", "landing.js"):
        assert needle in landing
    assert "—" not in landing and "—" not in (server.STATIC / "landing.js").read_text(encoding="utf-8")
