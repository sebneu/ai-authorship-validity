#!/usr/bin/env python3
"""Figures for the results sections.

Two, both answering a question the tables answer more precisely but less quickly.

    f_auroc.pdf       where every instrument sits relative to chance, in one view
    f_automation.pdf  what the instruments respond to when nothing in the text is
                      language-model output

Written as vector PDF at the column width of the Springer template.

Usage:
    python figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VALIDITY = ROOT / "data" / "processed" / "validity"
OUT = ROOT / "paper" / "figures"

LABELS = {
    "binoculars": "Binoculars",
    "fast_detect_gpt": "Fast-DetectGPT",
    "detect_code_gpt": "DetectCodeGPT",
    "heuristics": "Formatting rules",
    "selfadmission": "Keyword scan",
    "llm_judge": "LLM judge",
    "fingerprint": "Supervised ceiling",
}
ORDER = ["binoculars", "fast_detect_gpt", "detect_code_gpt", "selfadmission",
         "heuristics", "llm_judge", "fingerprint"]
GENRE_LABEL = {
    "commit_message": "Commit msg", "pr_body": "PR body", "issue_body": "Issue",
    "comment": "Comment", "diff": "Diff",
}
MARKERS = {"commit_message": "o", "pr_body": "s", "issue_body": "^",
           "comment": "D", "diff": "v"}

plt.rcParams.update({
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def auroc_figure(metrics: pd.DataFrame) -> None:
    """One row per instrument, one marker per genre, chance drawn as a line.

    A dot plot rather than bars: the quantity has a meaningful origin at 0.5 rather than
    at zero, and bars from zero would make a detector at chance look like most of a
    detector that works.
    """
    present = [d for d in ORDER if d in set(metrics.detector)]
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(present) + 1.1))

    ax.axvline(0.5, color="0.35", lw=1, zorder=1)
    ax.text(0.5, len(present) - 0.35, " chance", color="0.35", va="bottom", fontsize=7)

    for y, det in enumerate(present):
        sub = metrics[metrics.detector == det]
        ax.plot([sub.auroc_n1.min(), sub.auroc_n1.max()], [y, y],
                color="0.8", lw=3, solid_capstyle="round", zorder=2)
        for r in sub.itertuples():
            ax.scatter(r.auroc_n1, y, marker=MARKERS.get(r.genre, "o"), s=26,
                       facecolor="white", edgecolor="0.15", linewidth=0.9, zorder=3)

    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([LABELS[d] for d in present])
    ax.set_ylim(-0.6, len(present) - 0.4)
    ax.set_xlabel("Area under the ROC curve, against pre-ChatGPT text")
    ax.set_xlim(0.25, 1.0)
    ax.grid(axis="x", color="0.92", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], marker=m, linestyle="", markerfacecolor="white",
                          markeredgecolor="0.15", markersize=5, label=GENRE_LABEL[g])
               for g, m in MARKERS.items()]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2,
              handletextpad=0.3, columnspacing=1.0, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "f_auroc.pdf", bbox_inches="tight")
    plt.close(fig)


def automation_figure(automation: pd.DataFrame) -> None:
    """Paired bars: pre-LLM human text against pre-LLM bot text, per instrument."""
    pooled = (automation.groupby("detector")[["fpr_human", "fpr_bots"]]
              .mean().reindex([d for d in ORDER if d in set(automation.detector)]).dropna())
    y = np.arange(len(pooled))
    height = 0.36

    fig, ax = plt.subplots(figsize=(5.2, 0.5 * len(pooled) + 1.0))
    ax.barh(y + height / 2, 100 * pooled.fpr_human, height, label="Pre-ChatGPT humans",
            color="0.75", edgecolor="0.3", linewidth=0.6)
    ax.barh(y - height / 2, 100 * pooled.fpr_bots, height, label="Pre-ChatGPT bots",
            color="0.25", edgecolor="0.15", linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[d] for d in pooled.index])
    ax.invert_yaxis()
    ax.set_xlabel("Artifacts flagged (%), averaged over genres")
    # Upper right: the long bar belongs to the LLM judge at the bottom of the axis, so
    # the lower corner is the one place the legend cannot go without covering data.
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_xlim(0, max(100 * pooled.fpr_bots.max() * 1.18, 10))
    ax.grid(axis="x", color="0.92", lw=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUT / "f_automation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_parquet(VALIDITY / "rq1_metrics.parquet")
    automation = pd.read_parquet(VALIDITY / "rq2_automation.parquet")

    # The judge's own false positive rate is fixed at the target by calibration, so the
    # comparison bar is the achieved rate rather than the nominal one.
    auroc_figure(metrics)
    automation_figure(automation)
    print(f"wrote f_auroc.pdf and f_automation.pdf to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
