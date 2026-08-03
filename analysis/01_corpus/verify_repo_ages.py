#!/usr/bin/env python3
"""Determine which AIDev repositories have a pre-LLM history.

Resolves open item 1 in this directory's README: the matched negative set N1 can only
be drawn from repositories that existed before the cutoff, and a 25-repository sample
put that at roughly 71%. This measures it over the whole frame and writes the eligible
repository list that build_negatives.py consumes.

Authentication:
    Set GITHUB_TOKEN in your environment. Unauthenticated requests are capped at 60/h,
    which cannot finish this job; authenticated requests get 5,000/h.

        export GITHUB_TOKEN=$(cat ~/.config/crossd/github_token)

    Do not place the token in this repository or pass it on the command line, where it
    would land in your shell history.

Usage:
    python verify_repo_ages.py
    python verify_repo_ages.py --frame all_repository --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "aidev"
OUT_DIR = ROOT / "data" / "processed"

LLM_CUTOFF = datetime(2022, 11, 30, tzinfo=UTC)

API = "https://api.github.com/repos/"


class RateLimiter:
    """Pause all workers when GitHub says the budget is spent."""

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


def fetch(full_name: str, token: str | None, limiter: RateLimiter) -> dict:
    """Return {full_name, created_at, ...} or {full_name, error}."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ai-authorship-validity/corpus",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for attempt in range(4):
        limiter.wait()
        req = urllib.request.Request(API + full_name, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            return {
                "full_name": full_name,
                "created_at": data["created_at"],
                "pushed_at": data.get("pushed_at"),
                "archived": data.get("archived", False),
                "fork": data.get("fork", False),
                "default_branch": data.get("default_branch"),
            }
        except urllib.error.HTTPError as exc:
            # 404 is a real answer: the repo was renamed, deleted or made private.
            if exc.code == 404:
                return {"full_name": full_name, "error": "not_found"}
            if exc.code in (403, 429):
                reset = exc.headers.get("X-RateLimit-Reset")
                limiter.pause_until(float(reset) + 2 if reset else time.time() + 60)
                continue
            if exc.code >= 500:
                time.sleep(2**attempt)
                continue
            return {"full_name": full_name, "error": f"http_{exc.code}"}
        except Exception as exc:  # noqa: BLE001 - transient network faults
            if attempt == 3:
                return {"full_name": full_name, "error": str(exc)[:80]}
            time.sleep(2**attempt)
    return {"full_name": full_name, "error": "retries_exhausted"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", default="repository", choices=["repository", "all_repository"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="stop after N repositories (for smoke tests)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "repo_ages.parquet")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print(
            "GITHUB_TOKEN is not set. Unauthenticated requests are capped at 60/hour,\n"
            "which is not enough to finish this job. Export a token and re-run;\n"
            "see this script's docstring.",
            file=sys.stderr,
        )
        return 2

    frame = pd.read_parquet(RAW / f"{args.frame}.parquet")
    names = frame.full_name.dropna().unique().tolist()
    if args.limit:
        names = names[: args.limit]
    print(f"{len(names):,} repositories from {args.frame}\n")

    limiter = RateLimiter()
    started = time.time()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(lambda n: fetch(n, token, limiter), names), 1):
            results.append(row)
            if i % 250 == 0 or i == len(names):
                rate = i / max(time.time() - started, 1)
                eta = (len(names) - i) / max(rate, 1e-9)
                print(f"  {i:,}/{len(names):,}  {rate:.1f}/s  eta {eta / 60:.0f}m")

    df = pd.DataFrame(results)
    resolved = df[df.get("created_at").notna()] if "created_at" in df else df.iloc[0:0]
    created = pd.to_datetime(resolved.created_at, format="mixed", utc=True)
    eligible = created < LLM_CUTOFF

    df["eligible_for_n1"] = False
    df.loc[resolved.index, "eligible_for_n1"] = eligible.values

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    errors = df.error.notna().sum() if "error" in df else 0
    pct = 100 * eligible.mean() if len(resolved) else 0.0
    print(f"\nresolved      : {len(resolved):,} of {len(df):,} ({errors:,} unresolvable)")
    print(f"pre-{LLM_CUTOFF.date()} : {int(eligible.sum()):,} ({pct:.1f}%) -- the N1 frame")
    if len(resolved):
        print(f"created range : {created.min().date()} -> {created.max().date()}")
        print("by year       :", created.dt.year.value_counts().sort_index().to_dict())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
