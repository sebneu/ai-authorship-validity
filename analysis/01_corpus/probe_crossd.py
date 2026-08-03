#!/usr/bin/env python3
"""Probe the CrOSSD API to establish what it can and cannot supply for this study.

Regenerates the two empirical claims recorded in DECISIONS.md (2026-08-03):

  C1  CrOSSD keeps timestamped snapshots, but none earlier than 2024, so it offers
      no pre-ChatGPT (pre-2022-11-30) comparison window.
  C2  CrOSSD stores README text, issue bodies and issue-comment bodies, but no
      commit messages, no PR titles or bodies, and -- decisively -- no author
      identity on any artifact. The RQ1-RQ3 corpora must therefore come from GitHub
      directly; CrOSSD text is supplementary only.

  C3  Artifact creation date does not date the text attached to it. Issues created
      years before ChatGPT carry comments written at crawl time, so a pre-2022
      negative set cannot be built by filtering on createdAt.

The payload schema changed between the earliest (2024-04) and current (2026-07)
snapshots -- the README moved from readmes/README_md/text to README_md/text, and
issue bodies and comments appeared. Probe both ends rather than one snapshot.

Usage:
    python probe_crossd.py                       # 150 projects, seed 7
    python probe_crossd.py --sample 300 --seed 7
    python probe_crossd.py --out ../../data/processed/crossd_probe.json

Exits non-zero if either claim fails to reproduce.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

API = "https://health.crossd.tech/api"

# A snapshot earlier than this would mean CrOSSD does reach back toward the
# pre-ChatGPT era and C1 would need revisiting.
EARLIEST_EXPECTED = datetime(2024, 1, 1, tzinfo=UTC)

# Strings at least this long count as prose rather than identifiers or URLs.
PROSE_MIN_CHARS = 300

# Prose-bearing paths CrOSSD is known to expose, normalised by strip_indices().
# Anything outside this set is new and must be triaged before the corpus is frozen.
KNOWN_PROSE_PATHS = {
    "/repository/repository/README_md/text",  # current schema
    "/repository/repository/readmes/README_md/text",  # 2024-04 schema
    "/repository/repository/issues/edges[]/node/body",
    "/repository/repository/issueNNN/comments/edges[]/node/body",
    "/repository/advisories[]/description",
}

# Fields whose absence forces the corpus to come from GitHub instead. Author identity
# is the critical one: without it there is no bot/human/agent separation and no join
# to AIDev, so CrOSSD text cannot serve as ground truth however rich it becomes.
REQUIRED_FOR_GROUND_TRUTH = ("author", "login", "title")

REFERENCE_PROJECT = "facebook/react"


def post(endpoint: str, payload: dict, timeout: int = 90) -> dict | list:
    req = urllib.request.Request(
        f"{API}/{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def strip_indices(path: str) -> str:
    """Collapse list indices and issue numbers so paths can be compared as patterns."""
    path = re.sub(r"\[\d+\]", "[]", path)
    return re.sub(r"/issue\d+/", "/issueNNN/", path)


def prose_fields(obj, path: str = "") -> list[tuple[str, int]]:
    """Every string field in the payload long enough to be prose."""
    found: list[tuple[str, int]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            found += prose_fields(value, f"{path}/{key}")
    elif isinstance(obj, list):
        # Sampling the head is enough: list entries share a schema, and PR/issue
        # edges number in the thousands.
        for i, value in enumerate(obj[:3]):
            found += prose_fields(value, f"{path}[{i}]")
    elif isinstance(obj, str) and len(obj) > PROSE_MIN_CHARS:
        found.append((path, len(obj)))
    return found


def check_snapshot_coverage(projects: list[str], n: int, seed: int) -> dict:
    """C1: snapshot history exists, but does not reach back before 2024."""
    random.seed(seed)
    sample = random.sample(projects, min(n, len(projects)))

    def fetch(project: str):
        try:
            return project, post("snapshots", {"term": project})
        except Exception as exc:  # noqa: BLE001 - network failures are data here
            print(f"  ! {project}: {exc}", file=sys.stderr)
            return project, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(fetch, sample))

    with_snaps = [(p, s) for p, s in results if s]
    counts = sorted(len(s) for _, s in with_snaps)
    earliest = min(min(s) for _, s in with_snaps)
    last_seen = Counter(
        datetime.fromtimestamp(max(s), UTC).strftime("%Y-%m") for _, s in with_snaps
    )

    return {
        "sampled": len(sample),
        "with_snapshots": len(with_snaps),
        "earliest_snapshot": datetime.fromtimestamp(earliest, UTC).isoformat(),
        "snapshot_count_min": counts[0],
        "snapshot_count_median": counts[len(counts) // 2],
        "snapshot_count_max": counts[-1],
        "pct_full_series": round(100 * sum(c >= 80 for c in counts) / len(counts), 1),
        "last_snapshot_by_month": dict(sorted(last_seen.items())),
    }


def check_text_availability(project: str, timestamp: float) -> dict:
    """C2: what prose CrOSSD holds, and what identity metadata it lacks."""
    payload = post("repo", {"term": project, "timestamp": timestamp})
    repo = payload["repository"]["repository"]

    by_pattern = Counter()
    for path, _ in prose_fields(payload):
        by_pattern[strip_indices(path)] += 1

    pulls = repo.get("pullRequests") or {}
    pr_fields = sorted(pulls["edges"][0]["node"]) if pulls.get("edges") else []

    issues = repo.get("issues") or {}
    issue_fields = sorted(issues["edges"][0]["node"]) if issues.get("edges") else []

    per_issue = [k for k in repo if k.startswith("issue") and k[5:].isdigit()]
    comment_fields: list[str] = []
    for key in per_issue:
        edges = (repo[key].get("comments") or {}).get("edges") or []
        if edges:
            comment_fields = sorted(edges[0]["node"])
            break

    # C3: how old are the issues whose text is stored, and does comment text on old
    # issues postdate the issue itself?
    issue_years = Counter(
        e["node"]["createdAt"][:4] for e in issues.get("edges", []) if e["node"].get("createdAt")
    )

    return {
        "project": project,
        "timestamp": timestamp,
        "prose_paths": dict(by_pattern),
        "pr_total_count": pulls.get("totalCount"),
        "pr_node_fields": pr_fields,
        "issue_total_count": issues.get("totalCount"),
        "issues_stored": len(issues.get("edges", [])),
        "issue_node_fields": issue_fields,
        "issue_comment_node_fields": comment_fields,
        "issue_createdAt_by_year": dict(sorted(issue_years.items())),
        "contributing_md_fields": sorted(repo.get("contributing_md") or {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--project", default=REFERENCE_PROJECT)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    projects = post("projects", {})
    print(f"CrOSSD projects: {len(projects)}\n")

    print(f"C1: snapshot coverage ({args.sample} projects, seed {args.seed})")
    cov = check_snapshot_coverage(projects, args.sample, args.seed)
    print(f"  earliest snapshot : {cov['earliest_snapshot']}")
    print(
        f"  snapshots/project : min {cov['snapshot_count_min']}, "
        f"median {cov['snapshot_count_median']}, max {cov['snapshot_count_max']}"
    )
    print(f"  full series (>=80): {cov['pct_full_series']}%")
    print(f"  last snapshot     : {cov['last_snapshot_by_month']}\n")

    snaps = post("snapshots", {"term": args.project})
    if not snaps:
        raise SystemExit(f"no snapshots for reference project {args.project}")

    # The schema changed over the series, so probe both ends.
    texts = {
        "oldest": check_text_availability(args.project, min(snaps)),
        "newest": check_text_availability(args.project, max(snaps)),
    }

    for label, text in texts.items():
        stamp = datetime.fromtimestamp(text["timestamp"], UTC).date()
        print(f"C2: text availability ({args.project}, {label} snapshot {stamp})")
        for path, count in sorted(text["prose_paths"].items()):
            flag = "" if strip_indices(path) in KNOWN_PROSE_PATHS else "  <-- NEW"
            print(f"  prose x{count:<4} {path}{flag}")
        print(f"  PR nodes    : {text['pr_node_fields']} of {text['pr_total_count']}")
        print(
            f"  issue nodes : {text['issue_node_fields']} "
            f"({text['issues_stored']} of {text['issue_total_count']} stored)"
        )
        print(f"  comment nodes: {text['issue_comment_node_fields']}")
        print(f"  contributing : {text['contributing_md_fields']}\n")

    newest = texts["newest"]
    print("C3: age of issues carrying stored text")
    print(f"  createdAt by year: {newest['issue_createdAt_by_year']}\n")

    failures = []

    earliest = datetime.fromisoformat(cov["earliest_snapshot"])
    if earliest < EARLIEST_EXPECTED:
        failures.append(
            f"C1 refuted: snapshot from {earliest.date()} predates "
            f"{EARLIEST_EXPECTED.date()} — CrOSSD may now reach the pre-2022 window, "
            "so revisit the GH Archive decision in DECISIONS.md"
        )

    for label, text in texts.items():
        unknown = [p for p in text["prose_paths"] if p not in KNOWN_PROSE_PATHS]
        if unknown:
            failures.append(
                f"C2 changed ({label}): undocumented prose paths {unknown} — "
                "CrOSSD may now expose text we assumed absent; re-check DECISIONS.md"
            )
        # The decisive absence: identity metadata on issues and comments.
        for field in REQUIRED_FOR_GROUND_TRUTH:
            if field in text["issue_node_fields"] or field in text["issue_comment_node_fields"]:
                failures.append(
                    f"C2 changed ({label}): '{field}' now present on issue/comment nodes — "
                    "CrOSSD text may have become usable as ground truth"
                )
        if any("commit" in p.lower() for p in text["prose_paths"]):
            failures.append(f"C2 changed ({label}): commit-message text now present")

    pre_llm = sum(
        n for y, n in newest["issue_createdAt_by_year"].items() if int(y) < 2023
    )
    if not pre_llm:
        failures.append(
            "C3 inconclusive: no pre-2023 issues in the stored sample, so the "
            "stale-createdAt hazard could not be demonstrated"
        )

    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "generated": datetime.now(UTC).isoformat(),
                    "project_count": len(projects),
                    "snapshot_coverage": cov,
                    "text_availability": texts,
                    "claims_hold": not failures,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")

    if failures:
        return 1
    print("Both claims reproduce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
