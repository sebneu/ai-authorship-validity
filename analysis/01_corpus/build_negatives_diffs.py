#!/usr/bin/env python3
"""Collect pre-LLM code diffs (negative set N1) for the diff genre.

Runs after build_negatives_git.py, which has already established which commits predate
the cutoff. This pass only needs file content, so it clones with --filter=blob:none
(trees present, blobs fetched on demand) and asks git for the patches of a sample of
those commits. Sampling keeps the work bounded: the design needs a few thousand diffs
per genre, not the 1.5M commits N1 contains.

Diffs are emitted per file, matching how the positive set is built: DetectCodeGPT
expects contiguous code, and per-file keeps the language label meaningful.

Generated and vendored files are excluded. A lockfile is machine-written but not
model-written, so leaving it in N1 would put N3-like text inside the clean negative
set and depress the measured false positive rate.

Usage:
    python build_negatives_diffs.py --repos 5 --commits 5     # smoke test
    python build_negatives_diffs.py                           # default sample

Output: data/processed/corpus_v1/n1_diffs.parquet
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from _github import read_token
from build_negatives_git import CREDENTIAL_HELPER, RATE_LIMITED, run

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "corpus_v1"
COMMITS = OUT_DIR / "n1_commits.parquet"

# Machine-generated or third-party content. Present in history, but not human-written
# prose or code, and not model-written either.
EXCLUDE_PATH = re.compile(
    r"(^|/)(node_modules|vendor|third_party|thirdparty|externals|dist|build|"
    r"generated|__generated__|\.min\.|migrations)/|"
    r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock|"
    r"composer\.lock|Gemfile\.lock|go\.sum|\.pbxproj|\.snap)$|"
    r"\.(min\.js|min\.css|map|lock|svg|png|jpg|jpeg|gif|ico|pdf|woff2?|ttf|eot|"
    r"zip|gz|jar|so|dll|dylib|exe|bin|class|pyc)$",
    re.IGNORECASE,
)

DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$")

# Very large diffs are dominated by bulk edits and would skew the length distribution
# away from the positives, whose ninetieth percentile is around 7k characters.
MAX_DIFF_CHARS = 20_000


def split_patches(raw: str) -> list[tuple[str, str]]:
    """Split `git show` output into (filename, patch) pairs, hunks only."""
    out: list[tuple[str, str]] = []
    filename: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if filename and buf:
            body = "\n".join(buf).strip()
            if body:
                out.append((filename, body))

    for line in raw.splitlines():
        header = DIFF_HEADER.match(line)
        if header:
            flush()
            filename, buf = header.group(2), []
            continue
        if filename is None:
            continue
        # Keep the hunks; drop index/mode/---/+++ noise that carries no authorship.
        if line.startswith(("@@", "+", "-", " ")) and not line.startswith(("+++", "---")):
            buf.append(line)
    flush()
    return out


def collect(repo: str, shas: list[str], workdir: Path, token: str | None) -> dict:
    dest = workdir / repo.replace("/", "__")
    cmd = ["git"]
    if token:
        cmd += ["-c", f"credential.helper={CREDENTIAL_HELPER}"]
    cmd += [
        "clone",
        "--bare",
        "--filter=blob:none",  # trees kept, blobs fetched only for what we ask for
        "--no-tags",
        "--quiet",
        f"https://github.com/{repo}.git",
        str(dest),
    ]

    for attempt in range(3):
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        code, _, err = run(cmd)
        if code == 0:
            break
        if any(m in err for m in RATE_LIMITED) and attempt < 2:
            continue
        return {"repo": repo, "rows": [], "error": (err.strip().splitlines() or ["fail"])[-1][:110]}
    else:
        return {"repo": repo, "rows": [], "error": "clone_failed"}

    # One git show for all sampled commits: a single lazy-fetch negotiation rather
    # than one per commit.
    code, out, err = run(
        ["git", "show", "--format=%n__COMMIT__%H", "--unified=3", "--no-color",
         "--no-renames", "--diff-filter=AM", *shas],
        cwd=dest,
        timeout=1200,
    )
    rows: list[dict] = []
    if code == 0:
        for chunk in out.split("__COMMIT__")[1:]:
            sha, _, patch_text = chunk.partition("\n")
            for filename, patch in split_patches(patch_text):
                if EXCLUDE_PATH.search(filename):
                    continue
                if not (10 < len(patch) <= MAX_DIFF_CHARS):
                    continue
                rows.append(
                    {
                        "repo": repo,
                        "sha": sha.strip(),
                        "source_id": f"{sha.strip()}:{filename}",
                        "filename": filename,
                        "ext": Path(filename).suffix.lstrip(".").lower() or None,
                        "text": patch,
                        "n_chars": len(patch),
                        "genre": "diff",
                    }
                )

    shutil.rmtree(dest, ignore_errors=True)
    return {"repo": repo, "rows": rows, "error": None if code == 0 else "show_failed"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos", type=int, default=400, help="repositories to sample")
    ap.add_argument("--commits", type=int, default=20, help="commits sampled per repository")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "n1_diffs.parquet")
    args = ap.parse_args()

    if not COMMITS.exists():
        raise SystemExit(f"missing {COMMITS}\nrun: python build_negatives_git.py")

    commits = pd.read_parquet(COMMITS, columns=["repo", "sha", "is_bot", "is_merge"])
    # Merge commits carry no diff of their own; bot commits belong to N3.
    commits = commits[~commits.is_bot & ~commits.is_merge]

    rng = commits.repo.drop_duplicates().sample(
        min(args.repos, commits.repo.nunique()), random_state=args.seed
    )
    picked = commits[commits.repo.isin(rng)]
    # Shuffle once, then take the first N rows per repository: same effect as sampling
    # within each group, without the groupby-apply column-handling pitfalls.
    shuffled = picked.sample(frac=1.0, random_state=args.seed)
    per_repo = shuffled.groupby("repo").head(args.commits)
    plan = {repo: grp.sha.tolist() for repo, grp in per_repo.groupby("repo")}
    print(f"{len(plan):,} repositories, {sum(len(v) for v in plan.values()):,} commits\n")

    token = read_token()
    if token:
        os.environ["GH_CLONE_TOKEN"] = token
        print("cloning authenticated\n")

    workdir = Path(tempfile.mkdtemp(prefix="n1diff-"))
    rows: list[dict] = []
    failures = 0
    started = datetime.now(UTC)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect, r, s, workdir, token): r for r, s in plan.items()}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception:  # noqa: BLE001
                failures += 1
                continue
            failures += bool(result["error"])
            rows.extend(result["rows"])
            if i % 25 == 0 or i == len(plan):
                rate = i / max((datetime.now(UTC) - started).total_seconds(), 1)
                print(
                    f"  {i:,}/{len(plan):,} repos  {len(rows):,} diffs  "
                    f"eta {(len(plan) - i) / max(rate, 1e-9) / 60:.0f}m"
                )

    shutil.rmtree(workdir, ignore_errors=True)

    if not rows:
        print("\nno diffs collected", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\n{len(df):,} diffs from {df.repo.nunique():,} repositories ({failures} failed)")
    print(f"median {int(df.n_chars.median())} chars, p90 {int(df.n_chars.quantile(0.9))}")
    print("\ntop extensions:")
    print(df.ext.value_counts().head(12).to_string())
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
