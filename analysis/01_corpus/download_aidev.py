#!/usr/bin/env python3
"""Download the AIDev dataset and record exactly which version was used.

AIDev is a living dataset -- it has roughly doubled since the paper (arXiv:2507.15003)
reported 456k agentic PRs. Every download is therefore pinned to a commit sha and
recorded in a manifest so the corpus can be rebuilt bit-for-bit.

Usage:
    python download_aidev.py                      # pin to REVISION below
    python download_aidev.py --revision main      # take current head, then pin it
    python download_aidev.py --tables pull_request repository

Writes to data/raw/aidev/ and a manifest to
analysis/01_corpus/manifests/aidev_<revision>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "hao-li/AIDev"

# Pinned revision. Bump deliberately, never silently: re-running with a new revision
# changes every downstream count in the paper.
REVISION = "68ed5f4b80d27a9e057fc57567f38bd322ac73ec"

# Tables this study needs, and why. Ordered by role in the design.
TABLES = {
    # Positive ground truth -- agent-attributed text, by genre.
    "all_pull_request": "932k agent PRs: title + body (PR-description genre)",
    "pull_request": "33.6k curated agent PRs from 2,807 popular repos",
    "pr_commit_details": "commit messages + patches (commit + diff genres)",
    "pr_commits": "commit messages without patches (lighter alternative)",
    "pr_comments": "PR comment bodies; carries user_type for bot separation",
    "issue": "agent-linked issue titles + bodies (issue genre)",
    "related_issue": "issue <-> PR link table; the issue table has no agent column",
    # Contemporaneous human control.
    "human_pull_request": "6.6k human-authored PRs, same era and repos",
    # Stratification and identity.
    "repository": "curated repo metadata: language, stars, forks, license",
    "all_repository": "full repo frame (116k)",
    "user": "agent-repo user accounts",
    "all_user": "full user frame (72k)",
}

DEFAULT_DEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "aidev"
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--tables", nargs="*", default=sorted(TABLES))
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()

    unknown = set(args.tables) - set(TABLES)
    if unknown:
        print(f"unknown tables: {sorted(unknown)}", file=sys.stderr)
        print(f"available: {sorted(TABLES)}", file=sys.stderr)
        return 2

    api = HfApi()
    info = api.repo_info(REPO_ID, repo_type="dataset", revision=args.revision)
    resolved = info.sha
    if resolved != args.revision:
        print(f"note: '{args.revision}' resolves to {resolved}")
    print(f"AIDev @ {resolved} (modified {info.lastModified})\n")

    args.dest.mkdir(parents=True, exist_ok=True)
    entries = []

    for table in args.tables:
        filename = f"{table}.parquet"
        print(f"  {filename} -- {TABLES[table]}")
        local = Path(
            hf_hub_download(
                REPO_ID,
                filename,
                repo_type="dataset",
                revision=resolved,
                local_dir=args.dest,
            )
        )
        entries.append(
            {
                "table": table,
                "file": filename,
                "bytes": local.stat().st_size,
                "sha256": sha256(local),
                "purpose": TABLES[table],
            }
        )

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"aidev_{resolved[:12]}.json"

    # Merge rather than overwrite: downloading one extra table must not erase the
    # provenance record of the tables already fetched at this revision.
    merged = {e["table"]: e for e in entries}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        for entry in prior.get("files", []):
            merged.setdefault(entry["table"], entry)

    manifest_path.write_text(
        json.dumps(
            {
                "dataset": REPO_ID,
                "revision": resolved,
                "hub_last_modified": str(info.lastModified),
                "downloaded": datetime.now(UTC).isoformat(),
                "files": [merged[k] for k in sorted(merged)],
            },
            indent=2,
        )
    )

    total = sum(e["bytes"] for e in entries)
    print(f"\n{len(entries)} tables, {total / 1e9:.2f} GB -> {args.dest}")
    print(f"manifest -> {manifest_path}")
    if resolved != REVISION:
        print(
            f"\nREVISION in this script is {REVISION[:12]}, you downloaded "
            f"{resolved[:12]}. Update the constant and log it in DECISIONS.md."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
