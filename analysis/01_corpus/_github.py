"""Shared GitHub helpers: credential lookup and rate-limit handling.

Token resolution order:
  1. $GITHUB_TOKEN
  2. ~/.config/crossd/github_token   (first line, whitespace stripped)

The file exists so that long crawls can be started by any process without the token
living in a shell history or in this repository. Create it once:

    mkdir -p ~/.config/crossd
    printf '%s' 'ghp_...' > ~/.config/crossd/github_token
    chmod 600 ~/.config/crossd/github_token

The token is never logged, printed or written to any output file.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock

TOKEN_FILE = Path.home() / ".config" / "crossd" / "github_token"

API = "https://api.github.com"


def read_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token.strip()
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().splitlines()[0].strip()
        if token:
            mode = TOKEN_FILE.stat().st_mode & 0o077
            if mode:
                print(
                    f"warning: {TOKEN_FILE} is group/world readable; chmod 600 it",
                    file=sys.stderr,
                )
            return token
    return None


def require_token() -> str:
    token = read_token()
    if not token:
        raise SystemExit(
            "No GitHub token found.\n"
            "Unauthenticated requests are capped at 60/hour, which cannot finish this\n"
            "job. Provide one of:\n"
            "  export GITHUB_TOKEN=...\n"
            f"  printf '%s' '<token>' > {TOKEN_FILE} && chmod 600 {TOKEN_FILE}"
        )
    return token


class RateLimiter:
    """Shared backoff: when GitHub says stop, every worker stops."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._resume_at = 0.0

    def wait(self) -> None:
        while True:
            with self._lock:
                delay = self._resume_at - time.time()
            if delay <= 0:
                return
            print(f"  rate limited, sleeping {delay:.0f}s", file=sys.stderr)
            time.sleep(min(delay, 60))

    def pause_until(self, epoch: float) -> None:
        with self._lock:
            self._resume_at = max(self._resume_at, epoch)


def get(
    path: str,
    token: str,
    limiter: RateLimiter,
    params: dict | None = None,
    retries: int = 4,
) -> tuple[list | dict | None, str | None]:
    """GET an API path. Returns (payload, error); exactly one is None."""
    url = path if path.startswith("http") else f"{API}{path}"
    if params:
        query = urllib.parse.urlencode(params)  # type: ignore[name-defined]
        url = f"{url}?{query}"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "ai-authorship-validity/corpus",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    for attempt in range(retries):
        limiter.wait()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=45
            ) as resp:
                return json.load(resp), None
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                reset = exc.headers.get("X-RateLimit-Reset")
                limiter.pause_until(float(reset) + 2 if reset else time.time() + 60)
                continue
            if exc.code >= 500:
                time.sleep(2**attempt)
                continue
            # 404 (gone), 451 (withheld), 409 (empty repo) are answers, not failures.
            return None, f"http_{exc.code}"
        except Exception as exc:  # noqa: BLE001 - transient network faults
            if attempt == retries - 1:
                return None, str(exc)[:80]
            time.sleep(2**attempt)
    return None, "retries_exhausted"


import urllib.parse  # noqa: E402 - imported late to keep the docstring first
