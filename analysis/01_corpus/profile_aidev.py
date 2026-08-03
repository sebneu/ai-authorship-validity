#!/usr/bin/env python3
"""Profile the downloaded AIDev tables.

Regenerates every count quoted in this directory's README -- the agent mix, the text
length distributions that motivate length matching, the agent x genre cell counts that
set the sampling allocation, and the two facts that shape the negative sets (the
Co-authored-by rate and the bot share of PR comments).

Usage:
    python profile_aidev.py
    python profile_aidev.py --json ../../data/processed/aidev_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "aidev"

# Public release of ChatGPT: the boundary defining the pre-LLM negative set.
LLM_CUTOFF = "2022-11-30"


def load(table: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = RAW / f"{table}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}\nrun: python download_aidev.py")
    return pd.read_parquet(path, columns=columns)


def describe_len(series: pd.Series) -> dict:
    n = series.fillna("").str.len()
    return {
        "empty_pct": round(100 * (n == 0).mean(), 1),
        "median": int(n.median()),
        "mean": int(n.mean()),
        "p90": int(n.quantile(0.90)),
        "p99": int(n.quantile(0.99)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    out: dict = {"generated": datetime.now(UTC).isoformat(), "llm_cutoff": LLM_CUTOFF}

    # --- Agent mix and the temporal window -----------------------------------
    prs = load("all_pull_request", ["agent", "created_at", "body", "repo_url"])
    created = pd.to_datetime(prs.created_at, format="mixed", utc=True)
    counts = prs.agent.value_counts()

    out["all_pull_request"] = {
        "rows": len(prs),
        "repos": int(prs.repo_url.nunique()),
        "window": [str(created.min()), str(created.max())],
        "agents": counts.to_dict(),
        "top_agent_share_pct": round(100 * counts.iloc[0] / len(prs), 1),
        "body_len": describe_len(prs.body),
        "body_len_by_agent": {
            agent: describe_len(grp.body) for agent, grp in prs.groupby("agent")
        },
    }

    print(f"all_pull_request: {len(prs):,} PRs, {prs.repo_url.nunique():,} repos")
    print(f"  window: {created.min().date()} -> {created.max().date()}")
    print(f"  {counts.index[0]} share: {out['all_pull_request']['top_agent_share_pct']}%")
    for agent, stats in out["all_pull_request"]["body_len_by_agent"].items():
        print(f"  {agent:14} n={counts[agent]:>7,}  median body {stats['median']:>5} chars")

    # --- Commit messages: the shortest genre ---------------------------------
    commits = load("pr_commits", ["pr_id", "message"])
    trailer_rate = commits.message.fillna("").str.contains("Co-authored-by", case=False).mean()
    out["pr_commits"] = {
        "rows": len(commits),
        "message_len": describe_len(commits.message),
        "co_authored_by_pct": round(100 * trailer_rate, 1),
    }
    print(
        f"\npr_commits: {len(commits):,}  median message "
        f"{out['pr_commits']['message_len']['median']} chars, "
        f"Co-authored-by on {out['pr_commits']['co_authored_by_pct']}%"
    )

    # --- Comments are mostly bots --------------------------------------------
    comments = load("pr_comments", ["pr_id", "body", "user_type"])
    types = comments.user_type.value_counts()
    out["pr_comments"] = {
        "rows": len(comments),
        "user_type": types.to_dict(),
        "bot_pct": round(100 * types.get("Bot", 0) / len(comments), 1),
        "body_len_user_only": describe_len(comments.loc[comments.user_type == "User", "body"]),
    }
    print(f"pr_comments: {len(comments):,}, {out['pr_comments']['bot_pct']}% bot-authored")

    # --- Agent x genre cells, driving the allocation --------------------------
    curated = load("pull_request", ["id", "agent", "body"])
    agent_of = curated.set_index("id").agent
    details = load("pr_commit_details", ["pr_id", "patch"])

    cells = {
        "pr_body": curated.assign(n=curated.body.fillna("").str.len())
        .query("n > 0")
        .groupby("agent")
        .size(),
    }
    for genre, frame in [
        ("commit_msg", commits),
        ("diff", details),
        ("comment", comments[comments.user_type == "User"]),
    ]:
        joined = frame.groupby("pr_id").size().to_frame("c").join(agent_of, how="inner")
        cells[genre] = joined.groupby("agent").c.sum()

    table = pd.DataFrame(cells).fillna(0).astype(int)
    out["curated_cells"] = table.to_dict()
    out["curated_rows"] = len(curated)
    print(f"\ncurated pull_request: {len(curated):,}")
    print("agent x genre cell counts (curated join):")
    print(table.to_string())

    thin = table.min().min()
    print(f"\nthinnest cell: {thin:,} — allocation floor is 300 per cell")

    # --- Human control set ----------------------------------------------------
    human = load("human_pull_request", ["repo_url", "user", "body", "created_at"])
    hcreated = pd.to_datetime(human.created_at, format="mixed", utc=True)
    out["human_pull_request"] = {
        "rows": len(human),
        "repos": int(human.repo_url.nunique()),
        "users": int(human.user.nunique()),
        "window": [str(hcreated.min()), str(hcreated.max())],
        "body_len": describe_len(human.body),
    }
    print(
        f"\nhuman_pull_request (N2): {len(human):,} PRs, "
        f"{human.repo_url.nunique():,} repos, {human.user.nunique():,} users, "
        f"median body {out['human_pull_request']['body_len']['median']} chars"
    )

    # --- Stratification variables --------------------------------------------
    repos = load("repository")
    out["repository"] = {
        "rows": len(repos),
        "languages": repos.language.value_counts().head(12).to_dict(),
        "stars": {
            "median": int(repos.stars.median()),
            "p10": int(repos.stars.quantile(0.10)),
            "p90": int(repos.stars.quantile(0.90)),
        },
    }
    print(
        f"repository frame: {len(repos):,} repos, median {out['repository']['stars']['median']} stars"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
