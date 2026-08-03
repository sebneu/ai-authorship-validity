#!/usr/bin/env python3
"""Build the positive set: agent-attributed artifacts, by genre.

Implements section 3 of this directory's README. Two choices carry most of the weight:

Equal allocation, not proportional. AIDev is 87% one agent, so a proportionally
sampled corpus would measure that agent and report the number as general. Cells are
therefore capped at a common target, trading precision on the abundant agent for
usable precision on the scarce one.

Identical normalisation to the negative set. Trailers are stripped with the same
shared function used on pre-LLM commit messages (_text.strip_trailers). If they were
removed from one side only, a detector could separate the classes by reading a label.

Usage:
    python build_positives.py                    # 1,000 per agent x genre cell
    python build_positives.py --target 2000 --floor 300
    python build_positives.py --seed 7

Output: data/processed/corpus_v1/positives.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from _text import EXCLUDE_PATH, has_boilerplate, strip_trailers

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "aidev"
OUT_DIR = ROOT / "data" / "processed" / "corpus_v1"

# Below this a text carries almost no signal and detector scores are dominated by
# tokenizer edge effects. Recorded rather than silently dropped: how detectors behave
# on very short input is part of what this study measures, but they cannot be scored
# at all below a floor.
MIN_CHARS = 10


def load(table: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = RAW / f"{table}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}\nrun: python download_aidev.py")
    return pd.read_parquet(path, columns=columns)


def pr_bodies() -> pd.DataFrame:
    """PR descriptions from the full table, not the curated subset.

    The curated subset holds only 459 Claude Code PRs against 5,137 in the full table;
    using the full one keeps the scarcest cell above the floor.
    """
    df = load("all_pull_request", ["id", "agent", "title", "body", "repo_url", "created_at"])
    df = df[df.body.notna()]
    return pd.DataFrame(
        {
            "source_id": df.id.astype("int64"),
            "agent": df.agent,
            "genre": "pr_body",
            "text": df.body.str.strip(),
            "repo": df.repo_url.str.replace("https://github.com/", "", regex=False),
            "created_at": df.created_at,
        }
    )


def commit_messages(agent_of: pd.Series) -> pd.DataFrame:
    df = load("pr_commits", ["sha", "pr_id", "message", "author"])
    df = df[df.message.notna()].join(agent_of, on="pr_id", how="inner")
    stripped = df.message.map(strip_trailers)
    return pd.DataFrame(
        {
            "source_id": df.sha,
            "agent": df.agent,
            "genre": "commit_message",
            "text": [s[0] for s in stripped],
            "had_trailer": [s[1] for s in stripped],
            "repo": None,
            "created_at": None,
        }
    )


def diffs(agent_of: pd.Series) -> pd.DataFrame:
    """One row per changed file: DetectCodeGPT expects contiguous code, not a
    multi-file patch, and per-file keeps language attribution meaningful.

    Generated and vendored files are excluded with the same rule applied to the
    negative side. Lockfiles and minified bundles are machine-written on both sides,
    so keeping them here while dropping them there would leave the classes separable
    by file type rather than by authorship.
    """
    df = load("pr_commit_details", ["sha", "pr_id", "patch", "filename"])
    df = df[df.patch.notna()].join(agent_of, on="pr_id", how="inner")
    before = len(df)
    df = df[~df.filename.fillna("").str.contains(EXCLUDE_PATH)]
    print(f"    dropped {before - len(df):,} generated/vendored files")
    return pd.DataFrame(
        {
            "source_id": df.sha.astype(str) + ":" + df.filename.astype(str),
            "agent": df.agent,
            "genre": "diff",
            "text": df.patch,
            "filename": df.filename,
            "repo": None,
            "created_at": None,
        }
    )


def issue_bodies(agent_of: pd.Series) -> pd.DataFrame:
    df = load("issue", ["id", "body", "created_at", "html_url"])
    df = df[df.body.notna()].copy()
    # The issue table has no agent column; attribute via the linking table.
    link = load("related_issue")
    join_col = next((c for c in link.columns if "issue" in c.lower()), None)
    pr_col = next((c for c in link.columns if "pr" in c.lower()), None)
    if join_col and pr_col:
        link = link[[join_col, pr_col]].dropna()
        # issue_id arrives as float; ids must be integers on both sides or the map
        # silently matches nothing.
        link[join_col] = link[join_col].astype("int64")
        link[pr_col] = link[pr_col].astype("int64")
        link["agent"] = link[pr_col].map(agent_of)
        mapping = link.dropna(subset=["agent"]).drop_duplicates(join_col).set_index(join_col).agent
        df["agent"] = df.id.astype("int64").map(mapping)
    else:
        df["agent"] = pd.NA
    df = df[df.agent.notna()]
    return pd.DataFrame(
        {
            "source_id": df.id.astype("int64"),
            "agent": df.agent,
            "genre": "issue_body",
            "text": df.body.str.strip(),
            "repo": None,
            "created_at": df.created_at,
        }
    )


def comments(agent_of: pd.Series) -> pd.DataFrame:
    """Human-authored comments only: 70% of comments on agent PRs are bot-authored,
    so an unfiltered genre would measure CI bots rather than agents."""
    df = load("pr_comments", ["id", "pr_id", "body", "user_type", "created_at"])
    df = df[(df.body.notna()) & (df.user_type == "User")]
    df = df.join(agent_of, on="pr_id", how="inner")
    return pd.DataFrame(
        {
            "source_id": df.id.astype("int64"),
            "agent": df.agent,
            "genre": "comment",
            "text": df.body.str.strip(),
            "repo": None,
            "created_at": df.created_at,
        }
    )


def allocate(df: pd.DataFrame, target: int, floor: int, seed: int) -> tuple[pd.DataFrame, list]:
    """Equal allocation per (agent, genre) cell, with a reported shortfall."""
    kept, report = [], []
    for (agent, genre), cell in df.groupby(["agent", "genre"], sort=True):
        available = len(cell)
        take = min(target, available)
        sampled = cell.sample(take, random_state=seed) if take < available else cell
        kept.append(sampled)
        report.append(
            {
                "agent": agent,
                "genre": genre,
                "available": available,
                "sampled": take,
                "below_floor": available < floor,
            }
        )
    return pd.concat(kept, ignore_index=True), report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=1000, help="texts per agent x genre cell")
    ap.add_argument("--floor", type=int, default=300, help="warn below this many available")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "positives.parquet")
    args = ap.parse_args()

    curated = load("pull_request", ["id", "agent"])
    agent_of = curated.set_index("id").agent

    frames = []
    for name, builder in [
        ("pr_body", lambda: pr_bodies()),
        ("commit_message", lambda: commit_messages(agent_of)),
        ("diff", lambda: diffs(agent_of)),
        ("issue_body", lambda: issue_bodies(agent_of)),
        ("comment", lambda: comments(agent_of)),
    ]:
        frame = builder()
        print(f"  {name:15} {len(frame):>9,} candidates")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    # Genres identify artifacts differently -- numeric ids for PRs, issues and
    # comments, SHAs for commits, "sha:filename" for diffs -- so the column is text.
    df["source_id"] = df.source_id.astype(str)
    df["text"] = df.text.fillna("").str.strip()
    df["n_chars"] = df.text.str.len()

    too_short = int((df.n_chars < MIN_CHARS).sum())
    df = df[df.n_chars >= MIN_CHARS]
    df["n_lines"] = df.text.str.count("\n") + 1
    df["has_boilerplate"] = df.text.map(has_boilerplate)

    print(f"\n{len(df):,} candidates after dropping {too_short:,} under {MIN_CHARS} chars")
    print("\navailable per cell:")
    print(df.groupby(["genre", "agent"]).size().unstack(fill_value=0).to_string())

    sampled, report = allocate(df, args.target, args.floor, args.seed)
    sampled["label"] = "positive"
    sampled["set"] = "P"

    thin = [r for r in report if r["below_floor"]]
    if thin:
        print(f"\n{len(thin)} cell(s) below the floor of {args.floor}:")
        for r in thin:
            print(f"  {r['agent']} / {r['genre']}: {r['available']:,} available")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(args.out, index=False)

    manifest = args.out.with_name("positives_manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "generated": datetime.now(UTC).isoformat(),
                "seed": args.seed,
                "target_per_cell": args.target,
                "floor": args.floor,
                "min_chars": MIN_CHARS,
                "rows": len(sampled),
                "cells": report,
            },
            indent=2,
        )
    )

    print(f"\nsampled {len(sampled):,} texts")
    print(sampled.groupby(["genre", "agent"]).size().unstack(fill_value=0).to_string())
    print("\nmedian chars by genre:")
    for genre, grp in sampled.groupby("genre"):
        print(f"  {genre:15} {int(grp.n_chars.median()):>6}")
    print(f"\nwrote {args.out}\n      {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
