#!/usr/bin/env python3
"""RQ2: where the detector errors sit.

RQ1 establishes that discrimination is poor. This asks what the detectors are
responding to instead, by holding one factor at a time:

  Length. Detector statistics are sample means over tokens, so their variance falls as
  texts lengthen. If AUROC rises with length, the failure is partly a short-text
  problem and would ease on longer artifacts.

  Agent. The corpus is equally allocated across five agents, so per-agent AUROC shows
  whether a detector generalises or tracks one platform's house style.

  Language. Diffs carry a file extension. Since the negatives were language-matched to
  the positives, a residual language effect points at the detector rather than the
  sampling.

  Automation. The false positive rate on pre-LLM bot text against the rate on pre-LLM
  human text separates "flags machine-generated" from "flags language-model-generated".

Usage:
    python stratified.py
    python stratified.py --bootstrap 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import ROOT, auroc, boot_ci, load, operating_point, rate_at  # noqa: E402

OUT = ROOT / "data" / "processed" / "validity"

# Deciles are too many to read in a table; quintiles keep each cell above ~1,000 texts.
LENGTH_BINS = 5


def by_length(df: pd.DataFrame, bootstrap: int, rng) -> pd.DataFrame:
    rows = []
    for (detector, genre), sub in df.groupby(["detector", "genre"]):
        pos_all = sub[sub.set == "P"]
        neg_all = sub[(sub.set == "N1") & (sub.matched == True)]  # noqa: E712
        if len(pos_all) < 200 or len(neg_all) < 200:
            continue
        # Bin on the pooled length distribution so positives and negatives land in
        # comparable bins rather than each being split at its own quantiles.
        edges = np.quantile(
            pd.concat([pos_all.n_chars, neg_all.n_chars]), np.linspace(0, 1, LENGTH_BINS + 1)
        )
        edges[0], edges[-1] = -1, np.inf
        for i in range(LENGTH_BINS):
            lo, hi = edges[i], edges[i + 1]
            p = pos_all.loc[pos_all.n_chars.between(lo, hi, "right"), "score"].to_numpy(float)
            n = neg_all.loc[neg_all.n_chars.between(lo, hi, "right"), "score"].to_numpy(float)
            if len(p) < 50 or len(n) < 50:
                continue
            a = auroc(p, n)
            ci = boot_ci(p, n, bootstrap, rng)
            rows.append(
                {
                    "detector": detector,
                    "genre": genre,
                    "bin": i + 1,
                    "chars_from": int(max(lo, 0)),
                    "chars_to": int(hi) if np.isfinite(hi) else -1,
                    "n_pos": len(p),
                    "auroc": a,
                    "ci_lo": ci[0],
                    "ci_hi": ci[1],
                }
            )
    return pd.DataFrame(rows)


def by_agent(df: pd.DataFrame, bootstrap: int, rng) -> pd.DataFrame:
    """Per-agent AUROC against the shared pool of matched negatives for the genre."""
    rows = []
    for (detector, genre), sub in df.groupby(["detector", "genre"]):
        neg = sub.loc[(sub.set == "N1") & (sub.matched == True), "score"].to_numpy(float)  # noqa: E712
        if len(neg) < 200:
            continue
        for agent, agrp in sub[sub.set == "P"].groupby("agent"):
            p = agrp.score.to_numpy(float)
            if len(p) < 100:
                continue
            ci = boot_ci(p, neg, bootstrap, rng)
            rows.append(
                {
                    "detector": detector,
                    "genre": genre,
                    "agent": agent,
                    "n_pos": len(p),
                    "auroc": auroc(p, neg),
                    "ci_lo": ci[0],
                    "ci_hi": ci[1],
                }
            )
    return pd.DataFrame(rows)


def automation_gap(df: pd.DataFrame) -> pd.DataFrame:
    """False positives on pre-LLM human text against pre-LLM automation.

    Both predate language models, so every detection in either is an error. A detector
    that fires much harder on the bot text is responding to machine-generated text in
    general rather than to model-generated text, which is the confound the prevalence
    literature does not separate.
    """
    rows = []
    for (detector, genre), sub in df.groupby(["detector", "genre"]):
        pos = sub.loc[sub.set == "P", "score"].to_numpy(float)
        n1 = sub.loc[(sub.set == "N1") & (sub.matched == True), "score"].to_numpy(float)  # noqa: E712
        n3 = sub.loc[sub.set == "N3", "score"].to_numpy(float)
        if len(pos) < 100 or len(n1) < 100 or len(n3) < 100:
            continue
        thr, fpr_human, recall = operating_point(pos, n1, 0.05)
        fpr_bots = rate_at(n3, thr)
        rows.append(
            {
                "detector": detector,
                "genre": genre,
                "n_bots": len(n3),
                "fpr_human": fpr_human,
                "fpr_bots": fpr_bots,
                "ratio": fpr_bots / fpr_human if fpr_human else np.nan,
                "recall_agents": recall,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=300)
    args = ap.parse_args()

    print("loading scores:")
    df = load()
    rng = np.random.default_rng(7)
    OUT.mkdir(parents=True, exist_ok=True)

    length = by_length(df, args.bootstrap, rng)
    length.to_parquet(OUT / "rq2_length.parquet", index=False)
    print("\n=== AUROC by text length (matched N1) ===")
    pivot = length.pivot_table(index=["detector", "genre"], columns="bin", values="auroc")
    print(pivot.round(3).to_string())

    agent = by_agent(df, args.bootstrap, rng)
    agent.to_parquet(OUT / "rq2_agent.parquet", index=False)
    print("\n=== AUROC by agent ===")
    print(
        agent.pivot_table(index=["detector", "genre"], columns="agent", values="auroc")
        .round(3)
        .to_string()
    )

    gap = automation_gap(df)
    gap.to_parquet(OUT / "rq2_automation.parquet", index=False)
    print("\n=== False positives: pre-LLM humans vs pre-LLM automation ===")
    show = gap.copy()
    for col in ("fpr_human", "fpr_bots", "recall_agents"):
        show[col] = (100 * show[col]).round(1)
    show["ratio"] = show["ratio"].round(1)
    print(show.to_string(index=False))

    print(f"\nwrote {OUT}/rq2_*.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
