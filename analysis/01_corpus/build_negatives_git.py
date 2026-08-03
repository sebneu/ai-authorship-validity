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
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

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

# Trailer lines are explicit declarations. They are split out and never scored:
# recovering a declaration is not detection, and leaving them in would let a detector
# "identify" AI authorship by reading a label.
TRAILER = re.compile(
    r"^(Co-authored-by|Signed-off-by|Reviewed-by|Acked-by|Tested-by|Helped-by|"
    r"Reported-by|Suggested-by|Generated-by|Assisted-by):",
    re.IGNORECASE | re.MULTILINE,
)

# Automation accounts, identified from the commit author. These become N3 (pre-LLM
# machine-generated text) rather than being discarded -- quantifying how often
# detectors flag them is the point of N3.
BOT_PATTERNS = re.compile(
    r"(\[bot\]|dependabot|renovate|greenkeeper|snyk-bot|imgbot|allcontributors|"
    r"github-actions|semantic-release|release-please|mergify|pyup|whitesource|"
    r"scala-steward|depfu|codecov|travis|appveyor|circleci)",
    re.IGNORECASE,
)

MERGE_SUBJECT = re.compile(r"^(Merge (pull request|branch|remote-tracking|commit)\b|Merge tag\b)")


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 900) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, errors="replace"
    )
    return proc.returncode, proc.stdout, proc.stderr


def clone(full_name: str, dest: Path) -> str | None:
    """Bare, commit-graph-only clone. Returns an error string or None."""
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    code, _, err = run(
        [
            "git",
            "clone",
            "--bare",
            "--filter=tree:0",
            "--no-tags",
            "--quiet",
            f"https://github.com/{full_name}.git",
            str(dest),
        ]
    )
    if code != 0:
        first = err.strip().splitlines()[-1] if err.strip() else f"exit_{code}"
        return first[:120]
    return None


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

        message = body.strip("\n")
        trailers = TRAILER.findall(message)
        # Keep only the text a detector would see: drop trailer lines entirely.
        scored = "\n".join(
            line for line in message.splitlines() if not TRAILER.match(line)
        ).strip()
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
                "had_trailer": bool(trailers),
                "genre": "commit_message",
            }
        )
    return rows


def collect(full_name: str, workdir: Path, max_commits: int, keep: bool) -> dict:
    dest = workdir / full_name.replace("/", "__")
    err = clone(full_name, dest)
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
    ap.add_argument("--out", type=Path, default=OUT_DIR / "n1_commits.parquet")
    args = ap.parse_args()

    if not AGES.exists():
        raise SystemExit(f"missing {AGES}\nrun: python verify_repo_ages.py")

    ages = pd.read_parquet(AGES)
    repos = ages[ages.eligible_for_n1].full_name.tolist()
    if args.limit:
        repos = repos[: args.limit]

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
            pool.submit(collect, r, workdir, args.max_commits, args.keep_clones): r
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

    if not all_rows:
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
