"""
The HTTP layer: `http.server` from the standard library, and nothing else.

No framework, deliberately. §7 forbids introducing one without being asked and
forbids a dependency that duplicates the stdlib; this server routes nine paths
and serves one file, which `ThreadingHTTPServer` does perfectly well. The whole
harness installs and runs its UI with no web dependency at all, which is worth
more than the convenience a framework would buy.

Threading matters for one specific reason: a run executes in a background
thread and the browser polls `/api/job/...` while it runs. A single-threaded
server would block those polls behind the request that started the run, and the
progress bar would sit at zero until the run finished.

**Bound to loopback by default.** This is a local tool that can spend money and
reads a trace store; it has no authentication because it is not meant to be
reachable. Binding elsewhere is possible but takes an explicit `--host`, and
prints a warning, because the default should not be the dangerous one.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import api
from . import case_studies as cs
from . import platform_api as plat
from . import profile_api as papi
from . import reports as rep
from . import scaffold as sc

STATIC = Path(__file__).parent / "static"
MAX_BODY = 1 << 20      # 1 MiB: extraction previews are text, not uploads
MAX_UPLOAD = 24 << 20   # 24 MiB: a JSONL dataset from the Evaluate screen (20 MB of text, JSON-escaped)


class _Handler(BaseHTTPRequestHandler):
    # Injected by `serve()`; per-process config rather than globals in the
    # module, so two servers in one process do not share state.
    store_path = "runs/traces"
    models_cfg: dict = {}
    run_cfg: dict = {}
    budget = 0.0
    quiet = False

    server_version = "llm-eval-harness"
    sys_version = ""

    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:                       # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        route, query = parsed.path, parse_qs(parsed.query)

        if not route.startswith("/api/"):
            return self._static(route)

        try:
            self._send_json(self._get_route(route, query))
        except api.ApiError as e:
            self._send_json({"error": str(e)}, status=e.status)
        except Exception as e:                      # noqa: BLE001
            self._send_json({"error": f"{type(e).__name__}: {e}"}, status=500)

    def _get_route(self, route: str, query: dict) -> dict:
        one = lambda k, d="": (query.get(k) or [d])[0]   # noqa: E731

        if route == "/api/benchmarks":
            return api.list_benchmarks()
        if route == "/api/models":
            return api.list_models(self.models_cfg)
        if route == "/api/profiles":
            return papi.list_profiles()
        if route == "/api/runs":
            # One list, both halves. "Which runs do I have?" is a single
            # question; answering it twice suggests two systems.
            return papi.all_runs(self.store_path)
        if route == "/api/bench-runs":
            return api.list_runs(self.store_path)
        if route == "/api/profile-results":
            return papi.profile_results(
                self.store_path, profile=one("profile"),
                run_id=one("run_id"), metric=one("metric", "accuracy"),
                allow_unpaired=one("allow_unpaired") == "1")
        if route == "/api/decide":
            return papi.decide(
                self.store_path, profile=one("profile"), run_id=one("run_id"),
                require=[r for r in query.get("require", []) if r],
                optimise=one("optimise", "cost_usd"),
                quality_metric=one("quality_metric", "accuracy"),
                qpd=int(one("qpd", "10000")))
        if route == "/api/gate":
            return papi.gate(
                self.store_path,
                baseline=_need(one("baseline"), "baseline"),
                candidate=_need(one("candidate"), "candidate"),
                metrics=[m for m in query.get("metric", []) if m] or ["accuracy"],
                tolerance=float(one("tolerance", "0.02")),
                strict=one("strict") == "1",
                allow_unpaired=one("allow_unpaired") == "1")
        if route == "/api/results":
            return api.results(self.store_path, _need(one("run_id"), "run_id"))
        if route == "/api/rows":
            return api.rows(self.store_path, _need(one("run_id"), "run_id"),
                            limit=int(one("limit", "100")))
        if route == "/api/manifest":
            return api.manifest(self.store_path, _need(one("run_id"), "run_id"))
        if route == "/api/validate":
            return plat.validate(_need(one("profile"), "profile"), self.models_cfg)
        if route == "/api/estimate":
            return plat.estimate(
                _need(one("profile"), "profile"),
                [m for m in query.get("model", []) if m],
                self.models_cfg, self.run_cfg,
                do_adapted=one("adapted") == "1")
        if route == "/api/probes":
            fr = {k: one(k) for k in
                  ("unanswerable", "noise", "injection", "paraphrase",
                   "positional") if one(k)}
            return plat.probe_preview(
                _need(one("profile"), "profile"),
                seed=int(one("seed", "0")), **fr)
        if route == "/api/arena":
            return plat.arena(self.store_path, run_id=one("run_id"),
                              profile=one("profile"))
        if route == "/api/providers":
            return plat.providers()
        if route == "/api/rag-status":
            return plat.rag_status(self.run_cfg)
        if route == "/api/scaffold-spec":
            return sc.describe(one("task", "classify"))
        if route == "/api/case-studies":
            return cs.list_case_studies(self.store_path)
        if route == "/api/case-study":
            return cs.get_case_study(self.store_path, _need(one("id"), "id"))
        if route == "/api/reports":
            return rep.list_reports(self.store_path)
        if route == "/api/report":
            return rep.get_report(self.store_path, _need(one("id"), "id"))
        if route == "/api/connections":
            return plat.connections(include_models=one("models") == "1")
        if route == "/api/job":
            return api.job_status(_need(one("job_id"), "job_id"))
        raise api.ApiError(f"No such endpoint: {route}", 404)

    # ------------------------------------------------------------------ #
    def do_POST(self) -> None:                      # noqa: N802 - stdlib API
        route = urlparse(self.path).path
        try:
            body = self._read_json(MAX_UPLOAD if route == "/api/upload-dataset" else MAX_BODY)
            if route == "/api/scaffold-preview":
                out = sc.preview(_need(body.get("name"), "name"), _need(body.get("task"), "task"),
                                 dict(body.get("options") or {}))
            elif route == "/api/scaffold":
                out = sc.scaffold(_need(body.get("name"), "name"), _need(body.get("task"), "task"),
                                  dict(body.get("options") or {}),
                                  with_samples=bool(body.get("with_samples", True)),
                                  overwrite=bool(body.get("overwrite")))
            elif route == "/api/upload-dataset":
                labels = body.get("labels") or ""
                if isinstance(labels, str):
                    labels = [x.strip() for x in labels.replace("\n", ",").split(",") if x.strip()]
                out = sc.write_dataset(_need(body.get("name"), "name"), _need(body.get("kind"), "kind"),
                                       str(body.get("text") or ""), overwrite=bool(body.get("overwrite")),
                                       task=str(body.get("task") or ""), labels=list(labels))
            elif route == "/api/extract":
                out = api.extract_preview(
                    _need(body.get("benchmark"), "benchmark"),
                    str(body.get("raw", "")))
            elif route == "/api/run":
                out = api.start_run(
                    benchmark=_need(body.get("benchmark"), "benchmark"),
                    models=list(body.get("models") or []),
                    limit=int(body.get("limit", 20)),
                    seed=int(body.get("seed", 1729)),
                    store_path=self.store_path,
                    models_cfg=self.models_cfg,
                    budget=float(body.get("budget", self.budget) or 0.0),
                    acknowledge_licence=bool(body.get("acknowledge_licence")))
            elif route == "/api/save-report":
                out = rep.build_report(
                    self.store_path,
                    run_ids=list(body.get("run_ids") or []),
                    title=str(body.get("title") or ""),
                    provider=str(body.get("provider") or ""),
                    notes=str(body.get("notes") or ""),
                    caveats=list(body.get("caveats") or []))
            elif route == "/api/delete-report":
                out = rep.delete_report(self.store_path,
                                        _need(body.get("id"), "id"))
            elif route == "/api/connection-check":
                # The key, if supplied, is used for this one request and
                # dropped. It is never persisted anywhere (§6).
                out = plat.connection_check(
                    _need(body.get("provider"), "provider"),
                    body.get("api_key") or None)
            elif route == "/api/models-yaml":
                out = plat.suggested_models_yaml(
                    _need(body.get("provider"), "provider"),
                    list(body.get("models") or []))
            elif route == "/api/run-profile":
                out = plat.start_profile_run(
                    profile=_need(body.get("profile"), "profile"),
                    models=list(body.get("models") or []),
                    store_path=self.store_path, models_cfg=self.models_cfg,
                    run_cfg=self.run_cfg,
                    budget=float(body.get("budget", self.budget) or 0.0),
                    passes=body.get("passes"))
            else:
                raise api.ApiError(f"No such endpoint: {route}", 404)
            self._send_json(out)
        except api.ApiError as e:
            self._send_json({"error": str(e)}, status=e.status)
        except Exception as e:                      # noqa: BLE001
            self._send_json({"error": f"{type(e).__name__}: {e}"}, status=500)

    # ------------------------------------------------------------------ #
    def _read_json(self, limit: int = MAX_BODY) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > limit:
            raise api.ApiError("Request body too large.", 413)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise api.ApiError(f"Malformed JSON body: {e}", 400) from e

    def _send_json(self, payload, status: int = 200) -> None:
        blob = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        # Belt and braces for a local tool: nothing here should ever be framed
        # or sniffed, and the trace store can hold model output from anywhere.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(blob)

    def _static(self, route: str) -> None:
        """Serve the single-page app.

        Resolved against STATIC and checked to still be inside it, so a
        `../../` in the URL cannot read the trace store or a .env.
        """
        rel = "index.html" if route in ("/", "") else route.lstrip("/")
        try:
            target = (STATIC / rel).resolve()
            target.relative_to(STATIC.resolve())
        except (ValueError, OSError):
            return self._send_json({"error": "Not found"}, status=404)
        if not target.is_file():
            return self._send_json({"error": "Not found"}, status=404)

        blob = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("X-Content-Type-Options", "nosniff")
        # A local tool read off disk on every request: caching buys nothing and
        # costs a confusing hard-refresh every time the page changes. Without
        # this, an edit to app.js keeps serving the previous version and the
        # bug you just fixed appears not to be fixed.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, fmt: str, *args) -> None:
        """One line per request, or silence.

        The stdlib default writes to stderr with a timestamp format nobody
        wants; polling a job twice a second would bury the startup banner.
        """
        if not self.quiet:
            super().log_message(fmt, *args)


def _need(value, name: str):
    if value in (None, ""):
        raise api.ApiError(f"Missing required parameter: {name}", 400)
    return value


def make_server(host: str = "127.0.0.1", port: int = 8000, *,
                store_path: str = "runs/traces", models_cfg: dict | None = None,
                run_cfg: dict | None = None,
                budget: float = 0.0, quiet: bool = False) -> ThreadingHTTPServer:
    """Build the server without starting it, the shape tests need."""
    handler = type("Handler", (_Handler,), {
        "store_path": store_path,
        "models_cfg": models_cfg or {},
        "run_cfg": run_cfg or {},
        "budget": budget,
        "quiet": quiet,
    })
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str = "127.0.0.1", port: int = 8000, *,
          store_path: str = "runs/traces", models_cfg: dict | None = None,
          run_cfg: dict | None = None, budget: float = 0.0,
          open_browser: bool = True, quiet: bool = False) -> int:
    httpd = make_server(host, port, store_path=store_path,
                        models_cfg=models_cfg, run_cfg=run_cfg,
                        budget=budget, quiet=quiet)
    url = f"http://{host}:{port}/"

    print(f"[web] {url}")
    print(f"[web] store: {store_path}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[web] WARNING: bound to {host}, not loopback. This server has "
              f"no authentication and can spend money. Anyone who can reach "
              f"this port can start a run.")
    print("[web] Ctrl-C to stop.")

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped.")
    finally:
        httpd.server_close()
    return 0
