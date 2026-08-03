#!/usr/bin/env python3
"""Collect pre-LLM commit messages (negative set N1) from repository history.

Why git rather than the API: a commit message is dated by its committer timestamp and
cannot be edited without rewriting history, so the edit-contamination problem that
forces the double-timestamp filter on issue and PR bodies does not arise here. Cloning
is also unauthenticated and unmetered, where the API is neither.

Clones are bare and tree-filtered (--filter=tree:0), so only the commit graph is
fetched -- megabytes per repository rather than gigabytes. Diffs need file content and
are collected separately, on the sampled commits only.

Usage:
    python build_negatives_git.py --limit 5          # smoke test
    python build_negatives_git.py                    # full N1 frame
    python build_negatives_git.py --keep-clones      # retain clones for the diff pass

Output: data/processed/corpus_v1/n1_commits.parquet
"""

from __future__ import annotations

import argparse
import re
import os
import shutil
import subprocess
import time
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from _github import read_token
from _text import BOT_PATTERNS, strip_trailers

ROOT = Path(__file__).resolve().parents[2]
AGES = ROOT / "data" / "processed" / "repo_ages.parquet"
OUT_DIR = ROOT / "data" / "processed" / "corpus_v1"
CLONE_DIR = ROOT / "data" / "raw" / "clones"

# ChatGPT's public release. The boundary is exclusive and evaluated in UTC: commits
# dated on the day itself are ambiguous and are dropped.
LLM_CUTOFF = "2022-11-30"
CUTOFF_UTC = pd.Timestamp("2022-11-30T00:00:00Z")

# Field separator unlikely to occur in a commit message.
SEP = "\x1f"
REC = "\x1e"
LOG_FORMAT = SEP.join(["%H", "%cI", "%aI", "%an", "%ae", "%P", "%B"]) + REC

MERGE_SUBJECT = re.compile(r"^(Merge (pull request|branch|remote-tracking|commit)\b|Merge tag\b)")


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 900) -> tuple[int, str, str]:
    env = dict(os.environ)
    # Never let git block on an interactive credential prompt inside a worker thread.
    env["GIT_TERMINAL_PROMPT"] = "0"
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        errors="replace", env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


# Anonymous clones are rate limited per IP. Once the budget is spent GitHub starts
# demanding credentials, and git reports the misleading "expected flush after ref
# listing" rather than an auth error. Authenticating raises the ceiling enough to walk
# the whole frame; this pattern is how we detect the condition and back off.
RATE_LIMITED = ("expected flush after ref listing", "could not read Username")

# The token reaches git through a credential helper reading the environment, so it
# never appears in argv (visible to any user via ps) or in a URL.
CREDENTIAL_HELPER = (
    "!f() { echo username=x-access-token; echo \"password=$GH_CLONE_TOKEN\"; }; f"
)


def clone(full_name: str, dest: Path, token: str | None, attempts: int = 3) -> str | None:
    """Bare, commit-graph-only clone. Returns an error string or None."""
    cmd = ["git"]
    if token:
        cmd += ["-c", f"credential.helper={CREDENTIAL_HELPER}"]
    cmd += [
        "clone",
        "--bare",
        "--filter=tree:0",
        "--no-tags",
        "--quiet",
        f"https://github.com/{full_name}.git",
        str(dest),
    ]

    for attempt in range(attempts):
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        code, _, err = run(cmd)
        if code == 0:
            return None
        text = err.strip()
        if any(marker in text for marker in RATE_LIMITED) and attempt < attempts - 1:
            time.sleep(5 * (attempt + 1))
            continue
        return (text.splitlines()[-1] if text else f"exit_{code}")[:120]
    return "rate_limited"


def parse_log(raw: str, full_name: str) -> list[dict]:
    rows = []
    for record in raw.split(REC):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split(SEP)
        if len(parts) != 7:
            continue
        sha, cdate, adate, name, email, parents, body = parts

        scored, had_trailer = strip_trailers(body.strip("\n"))
        subject = scored.splitlines()[0] if scored.splitlines() else ""

        author = f"{name} <{email}>"
        rows.append(
            {
                "repo": full_name,
                "sha": sha,
                "committed_at": cdate,
                "authored_at": adate,
                "author": author,
                "text": scored,
                "n_chars": len(scored),
                "n_lines": len(scored.splitlines()),
                "is_merge": bool(len(parents.split()) > 1 or MERGE_SUBJECT.match(subject)),
                "is_bot": bool(BOT_PATTERNS.search(author)),
                "had_trailer": had_trailer,
                "genre": "commit_message",
            }
        )
    return rows


def collect(full_name: str, workdir: Path, max_commits: int, keep: bool, token: str | None) -> dict:
    dest = workdir / full_name.replace("/", "__")
    err = clone(full_name, dest, token)
    if err:
        return {"repo": full_name, "error": err, "rows": []}

    # --before uses committer date, which is what we want: it is when the commit
    # entered history, not when the work was originally authored (rebases move the
    # latter forward but not the former).
    code, out, log_err = run(
        [
            "git",
            "log",
            "--all",
            f"--before={LLM_CUTOFF}",
            f"--max-count={max_commits}",
            f"--pretty=format:{LOG_FORMAT}",
        ],
        cwd=dest,
    )
    rows = parse_log(out, full_name) if code == 0 else []
    error = None if code == 0 else (log_err.strip()[:120] or f"log_exit_{code}")

    if not keep:
        shutil.rmtree(dest, ignore_errors=True)
    return {"repo": full_name, "error": error, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="only the first N eligible repositories")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--max-commits",
        type=int,
        default=2000,
        help="cap per repository so large projects do not dominate the corpus",
    )
    ap.add_argument("--keep-clones", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip repositories already in the output and append")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "n1_commits.parquet")
    args = ap.parse_args()

    if not AGES.exists():
        raise SystemExit(f"missing {AGES}\nrun: python verify_repo_ages.py")

    ages = pd.read_parquet(AGES)
    repos = ages[ages.eligible_for_n1].full_name.tolist()
    if args.limit:
        repos = repos[: args.limit]

    # Authenticated clones get a far higher ceiling than anonymous ones, which run out
    # after roughly 1,200 repositories and then fail as confusing auth errors.
    token = read_token()
    if token:
        os.environ["GH_CLONE_TOKEN"] = token
        print("cloning authenticated")
    else:
        print("cloning anonymously -- expect rate limiting after ~1,200 repositories")

    existing = pd.DataFrame()
    if args.resume and args.out.exists():
        existing = pd.read_parquet(args.out)
        have = set(existing.repo.unique())
        repos = [r for r in repos if r not in have]
        print(f"resuming: {len(have):,} repositories already collected, {len(repos):,} to go")

    if "identity_ok" in ages and ages.identity_ok.notna().any():
        print("repository identity verified by GitHub id")
    else:
        print(
            "note: repo_ages.parquet predates the identity check -- re-run\n"
            "      verify_repo_ages.py to confirm names still point at AIDev's repos"
        )

    workdir = CLONE_DIR if args.keep_clones else Path(tempfile.mkdtemp(prefix="n1-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(repos):,} repositories -> {workdir}\n")

    all_rows: list[dict] = []
    failures: list[dict] = []
    started = datetime.now(UTC)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(collect, r, workdir, args.max_commits, args.keep_clones, token): r
            for r in repos
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - clone timeouts etc.
                failures.append({"repo": futures[future], "error": str(exc)[:120]})
                continue
            if result["error"]:
                failures.append({"repo": result["repo"], "error": result["error"]})
            all_rows.extend(result["rows"])
            if i % 25 == 0 or i == len(repos):
                elapsed = (datetime.now(UTC) - started).total_seconds()
                rate = i / max(elapsed, 1)
                print(
                    f"  {i:,}/{len(repos):,} repos  {len(all_rows):,} commits  "
                    f"{rate:.1f} repo/s  eta {(len(repos) - i) / max(rate, 1e-9) / 60:.0f}m"
                )

    if not args.keep_clones:
        shutil.rmtree(workdir, ignore_errors=True)

    if not all_rows and not len(existing):
        print("\nno commits collected", file=sys.stderr)
        for f in failures[:10]:
            print(f"  {f['repo']}: {f['error']}", file=sys.stderr)
        return 1

    df = pd.DataFrame(all_rows)
    df = df[df.n_chars > 0]

    # git's --before compares each commit against its own local timezone, so commits
    # on the cutoff day leak through. The purity of N1 is the whole point of this set,
    # so the boundary is re-applied here in UTC and both timestamps must clear it:
    # clock skew can leave a commit authored after it was committed.
    before = len(df)
    committed = pd.to_datetime(df.committed_at, format="mixed", utc=True)
    authored = pd.to_datetime(df.authored_at, format="mixed", utc=True)
    df = df[(committed < CUTOFF_UTC) & (authored < CUTOFF_UTC)]
    leaked = before - len(df)
    if leaked:
        print(f"\ndropped {leaked:,} commits at or after {CUTOFF_UTC.date()} (UTC boundary)")

    if len(existing):
        df = pd.concat([existing, df], ignore_index=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    if failures:
        fail_path = args.out.with_name(args.out.stem + "_failures.parquet")
        pd.DataFrame(failures).to_parquet(fail_path, index=False)

    human = df[~df.is_bot & ~df.is_merge]
    print(f"\ncollected {len(df):,} commit messages from {df.repo.nunique():,} repositories")
    print(f"  bot-authored (N3) : {int(df.is_bot.sum()):,}")
    print(f"  merge commits     : {int(df.is_merge.sum()):,}")
    print(f"  N1 after both     : {len(human):,}")
    if len(human):
        print(
            f"  length: median {int(human.n_chars.median())} chars, "
            f"p90 {int(human.n_chars.quantile(0.9))}"
        )
    print(f"  had trailers      : {int(df.had_trailer.sum()):,}")
    print(f"  failures          : {len(failures):,}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
