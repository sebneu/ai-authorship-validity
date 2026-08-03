"""Read configuration from the repository .env.

Deliberately not python-dotenv: the format we need is three lines of KEY=value, and a
dependency that silently interpolates or exports is not worth it for that. Values
already present in the environment win, so a shell export can override the file.

.env is gitignored. .env.example documents the keys and holds no values.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

_loaded = False


def load(path: Path | None = None) -> None:
    global _loaded
    if _loaded:
        return
    target = path or ENV_FILE
    if target.exists():
        mode = target.stat().st_mode & 0o077
        if mode:
            print(f"warning: {target} is group/world readable; chmod 600 it")
        for line in target.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if value and key not in os.environ:
                os.environ[key] = value
    _loaded = True


def get(key: str, default: str | None = None) -> str | None:
    load()
    return os.environ.get(key, default)


def require(key: str) -> str:
    value = get(key)
    if not value:
        raise SystemExit(
            f"{key} is not set.\n"
            f"Copy .env.example to .env and fill it in, or export {key}."
        )
    return value
