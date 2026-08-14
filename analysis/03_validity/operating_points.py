#!/usr/bin/env python3
"""What happens at the thresholds the literature actually publishes.

Every other result here calibrates a threshold on our own negatives, which answers how
well an instrument *could* work. That is not how these instruments are deployed. A
prevalence study picks a published operating point and counts what lands above it, so
the question this script asks is what that published point does to text known to
predate language models.

\\citet{ji2026exploratory} give the only operating points published for the
software-engineering domain (their Table 3), derived from a dataset of LLM-generated and
human-written code comments, and they apply them to repository artifacts. Two of the
eight are for instruments in our harness on a comparable scale:

    Binoculars      -1.00
    Fast-DetectGPT   1.18

The other six (DetectGPT, Log-Likelihood, Log-Rank, Rank, LRR, Entropy) are statistics
we do not run as separate adapters, so no threshold of ours corresponds to them.

**Scale and sign.** A published threshold is only meaningful against the same statistic
on the same scale. Binoculars is a ratio of perplexity to cross-perplexity and our
adapter returns its negation, so their -1.00 is our score >= -1.00, which is the ratio
at or below 1.00; we score under the falcon-7b pair of the original publication, so the
scale is the canonical one. Fast-DetectGPT's conditional curvature is standardised, and
our adapter returns it unnegated, so 1.18 transfers directly in sign and approximately
in scale -- approximately, because the statistic depends on the scoring model, and that
dependence is part of what transporting a threshold costs.

Usage:
    python operating_points.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import ROOT, load  # noqa: E402

OUT = ROOT / "data" / "processed" / "validity"
TEX = ROOT / "paper" / "generated" / "t_published.tex"

# Thresholds as published, on the publishing paper's own scale.
PUBLISHED = {
    "binoculars": -1.00,
    "fast_detect_gpt": 1.18,
}
LABELS = {"binoculars": "Binoculars", "fast_detect_gpt": "Fast-DetectGPT"}
GENRES = [
    ("commit_message", "Commit msg"),
    ("pr_body", "PR body"),
    ("issue_body", "Issue"),
    ("comment", "Comment"),
    ("diff", "Diff"),
]


def main() -> int:
    print("loading scores:")
    df = load()

    rows = []
    for detector, threshold in PUBLISHED.items():
        sub_all = df[df.detector == detector]
        if sub_all.empty:
            print(f"  {detector}: no scores, skipped")
            continue
        for genre, _ in GENRES:
            sub = sub_all[sub_all.genre == genre]
            n1 = sub.loc[(sub.set == "N1") & (sub.matched == True), "score"].to_numpy(float)  # noqa: E712
            pos = sub.loc[sub.set == "P", "score"].to_numpy(float)
            n3 = sub.loc[sub.set == "N3", "score"].to_numpy(float)
            if not len(n1) or not len(pos):
                continue
            rows.append(
                {
                    "detector": detector,
                    "genre": genre,
                    "threshold": threshold,
                    "fpr_prellm": float((n1 >= threshold).mean()),
                    "recall_agents": float((pos >= threshold).mean()),
                    "fpr_bots": float((n3 >= threshold).mean()) if len(n3) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    # Youden's J at the published point, which is what a correction would have to divide
    # by if a study tried to correct its own estimate.
    out["youden_j"] = out.recall_agents - out.fpr_prellm
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "rq1_published_points.parquet", index=False)

    print("\nat the published operating point (per cent)\n")
    show = out.copy()
    for col in ("fpr_prellm", "recall_agents", "fpr_bots"):
        show[col] = (100 * show[col]).round(1)
    show["youden_j"] = show.youden_j.round(3)
    print(show.to_string(index=False))

    write_table(out)
    print(f"\nwrote {OUT / 'rq1_published_points.parquet'} and {TEX}")
    return 0


def write_table(out: pd.DataFrame) -> None:
    body = []
    for detector, label in LABELS.items():
        sub = out[out.detector == detector]
        if sub.empty:
            continue
        worst = sub.fpr_prellm.max()
        cells = []
        for genre, _ in GENRES:
            cell = sub[sub.genre == genre]
            if cell.empty:
                cells.append("---")
                continue
            value = cell.fpr_prellm.iloc[0]
            text = f"{100 * value:.1f}"
            cells.append(rf"\textbf{{{text}}}" if value == worst else text)
        thr = PUBLISHED[detector]
        body.append(f"{label} ({thr:+.2f}) & " + " & ".join(cells) + r" \\")

    header = ["Detector (threshold)"] + [lbl for _, lbl in GENRES]
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text(rf"""% generated by analysis/03_validity/operating_points.py -- do not edit
\begin{{table}}[t]
\caption{{Share of pre-ChatGPT text flagged (\%) at the operating points published for
the software-engineering domain by \citet{{ji2026exploratory}}. Every one of these is a
false positive.}}
\label{{tab:published}}
\small
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}l{"r" * len(GENRES)}@{{}}}}
\toprule
{" & ".join(header)} \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular*}}

\vspace{{2pt}}
{{\footnotesize Thresholds are applied on the publishing paper's own scale; see
Section~\ref{{sec:groundtruth:harness}} for the sign and scoring-model mapping.}}
\end{{table}}
""")


if __name__ == "__main__":
    sys.exit(main())
