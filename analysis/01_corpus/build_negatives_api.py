#!/usr/bin/env python3
"""Collect pre-LLM issue bodies, PR descriptions and comments (negative set N1).

Unlike commit messages, these bodies are editable after the fact and issue threads keep
accruing comments for years. An artifact is therefore admitted only if BOTH its
creation and its last-modification timestamps precede the cutoff, which proves the text
has not been touched since. See this directory's README for why creation date alone is
not enough.

The crawl is resumable: each repository's result is appended to a JSONL as it lands, and
repositories already present are skipped on restart. A multi-hour crawl should not have
to start over because of a dropped connection.

Authentication: see _github.py. Unauthenticated runs cannot finish.

Usage:
    python build_negatives_api.py --limit 5        # smoke test
    python build_negatives_api.py                  # full N1 frame, resumable
    python build_negatives_api.py --with-comments  # also fetch comment threads
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from _github import RateLimiter, get, require_token

ROOT = Path(__file__).resolve().parents[2]
AGES = ROOT / "data" / "processed" / "repo_ages.parquet"
OUT_DIR = ROOT / "data" / "processed" / "corpus_v1"

CUTOFF = pd.Timestamp("2022-11-30T00:00:00Z")

BOT_MARKERS = ("[bot]", "dependabot", "renovate", "greenkeeper", "codecov", "github-actions")


def is_bot(user: dict | None) -> bool:
    if not user:
        return False
    if user.get("type") == "Bot":
        return True
    login = (user.get("login") or "").lower()
    return any(m in login for m in BOT_MARKERS)


def before_cutoff(*stamps: str | None) -> bool:
    """True only if every supplied timestamp exists and precedes the cutoff."""
    for stamp in stamps:
        if not stamp:
            return False
        if pd.Timestamp(stamp) >= CUTOFF:
            return False
    return True


def collect_repo(
    full_name: str,
    token: str,
    limiter: RateLimiter,
    max_pages: int,
    want_comments: bool,
) -> dict:
    """Issues and PRs created and last touched before the cutoff, oldest first."""
    rows: list[dict] = []
    errors: list[str] = []

    for page in range(1, max_pages + 1):
        payload, err = get(
            f"/repos/{full_name}/issues",
            token,
            limiter,
            params={
                "state": "all",
                "sort": "created",
                "direction": "asc",  # oldest first: the pre-cutoff era comes first
                "per_page": 100,
                "page": page,
            },
        )
        if err:
            errors.append(f"issues p{page}: {err}")
            break
        if not payload:
            break

        stop = False
        for item in payload:
            # Sorted ascending by creation, so once we pass the cutoff every later
            # page is post-cutoff too and the repository is done.
            if pd.Timestamp(item["created_at"]) >= CUTOFF:
                stop = True
                break
            if not before_cutoff(item.get("updated_at")):
                continue  # edited or commented on after the cutoff: not provably clean
            body = (item.get("body") or "").strip()
            if not body:
                continue
            rows.append(
                {
                    "repo": full_name,
                    "id": item["id"],
                    "number": item["number"],
                    "genre": "pr_body" if item.get("pull_request") else "issue_body",
                    "text": body,
                    "n_chars": len(body),
                    "created_at": item["created_at"],
                    "updated_at": item["updated_at"],
                    "author": (item.get("user") or {}).get("login"),
                    "is_bot": is_bot(item.get("user")),
                }
            )
        if stop or len(payload) < 100:
            break

    if want_comments and rows:
        for page in range(1, max_pages + 1):
            payload, err = get(
                f"/repos/{full_name}/issues/comments",
                token,
                limiter,
                params={"sort": "created", "direction": "asc", "per_page": 100, "page": page},
            )
            if err:
                errors.append(f"comments p{page}: {err}")
                break
            if not payload:
                break
            stop = False
            for c in payload:
                if pd.Timestamp(c["created_at"]) >= CUTOFF:
                    stop = True
                    break
                if not before_cutoff(c.get("updated_at")):
                    continue
                body = (c.get("body") or "").strip()
                if not body:
                    continue
                rows.append(
                    {
                        "repo": full_name,
                        "id": c["id"],
                        "number": None,
                        "genre": "comment",
                        "text": body,
                        "n_chars": len(body),
                        "created_at": c["created_at"],
                        "updated_at": c["updated_at"],
                        "author": (c.get("user") or {}).get("login"),
                        "is_bot": is_bot(c.get("user")),
                    }
                )
            if stop or len(payload) < 100:
                break

    return {"repo": full_name, "rows": rows, "errors": errors}


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["repo"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-pages", type=int, default=5, help="pages per endpoint per repo")
    ap.add_argument("--with-comments", action="store_true")
    ap.add_argument("--restart", action="store_true", help="ignore prior progress")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "n1_api.parquet")
    args = ap.parse_args()

    token = require_token()

    if not AGES.exists():
        raise SystemExit(f"missing {AGES}\nrun: python verify_repo_ages.py")
    ages = pd.read_parquet(AGES)
    repos = ages[ages.eligible_for_n1].full_name.tolist()
    if args.limit:
        repos = repos[: args.limit]

    progress = args.out.with_suffix(".jsonl")
    progress.parent.mkdir(parents=True, exist_ok=True)
    if args.restart and progress.exists():
        progress.unlink()

    done = load_done(progress)
    todo = [r for r in repos if r not in done]
    print(f"{len(repos):,} eligible repositories, {len(done):,} already done, {len(todo):,} to go\n")

    limiter = RateLimiter()
    started = datetime.now(UTC)
    collected = 0

    with progress.open("a") as sink, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(collect_repo, r, token, limiter, args.max_pages, args.with_comments): r
            for r in todo
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"repo": futures[future], "rows": [], "errors": [str(exc)[:100]]}
            sink.write(json.dumps(result) + "\n")
            sink.flush()  # a crash must not lose completed repositories
            collected += len(result["rows"])
            if i % 25 == 0 or i == len(todo):
                rate = i / max((datetime.now(UTC) - started).total_seconds(), 1)
                print(
                    f"  {i:,}/{len(todo):,} repos  {collected:,} artifacts  "
                    f"eta {(len(todo) - i) / max(rate, 1e-9) / 60:.0f}m"
                )

    rows, errors = [], []
    with progress.open() as fh:
        for line in fh:
            rec = json.loads(line)
            rows.extend(rec["rows"])
            errors.extend((rec["repo"], e) for e in rec["errors"])

    if not rows:
        print("\nno artifacts collected", file=sys.stderr)
        for repo, err in errors[:10]:
            print(f"  {repo}: {err}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)

    print(f"\ncollected {len(df):,} artifacts from {df.repo.nunique():,} repositories")
    print(df.groupby(["genre", "is_bot"]).size().to_string())
    human = df[~df.is_bot]
    print(f"\nN1 (human) : {len(human):,}   N3 (bot): {int(df.is_bot.sum()):,}")
    for genre, grp in human.groupby("genre"):
        print(f"  {genre:11} n={len(grp):>7,}  median {int(grp.n_chars.median()):>5} chars")
    print(f"errors: {len(errors):,}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
