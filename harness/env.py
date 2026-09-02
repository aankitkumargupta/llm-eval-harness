"""
Loading API keys from a `.env` file.

`python-dotenv` was a declared dependency and both the README and the
deployment guide told people to put their keys in `.env` — but nothing ever
called `load_dotenv()`. The file was read by nobody. Following the documented
setup produced "No API key for provider 'together'", with the key sitting right
there in a file the app never opened.

That is the worst shape a config bug takes: the instructions are wrong, the
error blames the user, and the fix is invisible.

Two rules, both deliberate:

  * **Real environment variables win.** A key exported in the shell, injected by
    Docker, or supplied by a CI secret store is more specific than a file
    checked into a working directory, and must not be silently overridden by a
    stale `.env` someone forgot about.
  * **A missing file, or a missing dotenv package, is not an error.** Plenty of
    deployments only ever use real environment variables. This is a
    convenience, so it fails quiet and returns what it did.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | Path = ".env", override: bool = False) -> list[str]:
    """Load `path` into the environment. Returns the names of variables set.

    `override=False` means an already-exported variable keeps its value, so the
    precedence is: real environment > .env file > provider default.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return []

    try:
        from dotenv import dotenv_values
    except ImportError:
        # Optional dependency. Fall back to a minimal parser rather than making
        # the whole app depend on it for a handful of KEY=value lines.
        values = _parse(p)
    else:
        values = {k: v for k, v in dotenv_values(p).items() if v is not None}

    applied: list[str] = []
    for key, value in values.items():
        if override or not os.environ.get(key):
            os.environ[key] = value
            applied.append(key)
    return applied


def _parse(path: Path) -> dict[str, str]:
    """A minimal KEY=value reader, for when python-dotenv is not installed.

    Handles comments, blank lines, `export` prefixes and surrounding quotes —
    the shapes a hand-written key file actually takes. Anything more exotic is
    what python-dotenv is for.
    """
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out
