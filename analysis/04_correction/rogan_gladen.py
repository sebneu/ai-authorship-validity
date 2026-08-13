#!/usr/bin/env python3
"""RQ3: what the measured error rates do to a published prevalence estimate.

Rogan and Gladen (1978) recover true prevalence from the share a fallible test flags:

    pi = (AP + Sp - 1) / (Se + Sp - 1)

with AP the apparent prevalence, Se sensitivity and Sp specificity. The denominator is
Youden's J. It is the whole story here. J is how much the test separates the classes at
its operating point, and the correction divides by it, so an instrument with J near zero
does not merely give an imprecise estimate: it gives one whose bounds run to the edges
of the unit interval whatever the observed rate.

The published estimates this paper audits report a share of artifacts flagged. That is
AP, not prevalence, and the gap between them is the quantity measured in RQ1 and RQ2.
Rather than pinning the analysis to one paper's headline number, the correction is
reported across a grid of apparent prevalences spanning the range those studies report.
The shape of the result does not depend on which number is chosen, which is the point.

Usage:
    python rogan_gladen.py
    python rogan_gladen.py --bootstrap 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "03_validity"))

from metrics import TARGET_FPR, load, operating_point  # noqa: E402

OUT = ROOT / "data" / "processed" / "correction"

# Apparent prevalences to correct. The measurement studies report flagged shares in the
# low tens of percent, so the grid brackets that with room on both sides.
AP_GRID = [0.05, 0.10, 0.20, 0.30, 0.40]


def rogan_gladen(ap: float | np.ndarray, se: float | np.ndarray, sp: float | np.ndarray):
    """Corrected prevalence, NaN where the test carries no information.

    Youden's J at or below zero means the test flags negatives at least as readily as
    positives. The estimator is then undefined rather than merely uncertain, and
    returning NaN keeps that visible instead of letting a clipped value through as if
    it were an estimate.
    """
    j = np.asarray(se) + np.asarray(sp) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = (np.asarray(ap) + np.asarray(sp) - 1) / j
    return np.where(j > 0, np.clip(pi, 0, 1), np.nan)


def main() -> int:
    ap_parse = argparse.ArgumentParser(description=__doc__)
    ap_parse.add_argument("--bootstrap", type=int, default=1000)
    ap_parse.add_argument("--out", type=Path, default=OUT)
    args = ap_parse.parse_args()

    print("loading scores:")
    df = load()
    rng = np.random.default_rng(7)

    rows = []
    for (detector, genre), sub in df.groupby(["detector", "genre"]):
        pos = sub.loc[sub.set == "P", "score"].to_numpy(float)
        neg = sub.loc[(sub.set == "N1") & (sub.matched == True), "score"].to_numpy(float)  # noqa: E712
        pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
        if len(pos) < 100 or len(neg) < 100:
            continue

        # The threshold is fixed once, from the full sample, and held across the
        # resamples. A detector in deployment has one threshold; refitting it inside
        # every bootstrap draw would report the accuracy of a procedure nobody runs.
        thr, fpr, recall = operating_point(pos, neg, TARGET_FPR)
        se, sp = recall, 1 - fpr

        draws_se = np.empty(args.bootstrap)
        draws_sp = np.empty(args.bootstrap)
        for b in range(args.bootstrap):
            draws_se[b] = (rng.choice(pos, len(pos), replace=True) >= thr).mean()
            draws_sp[b] = 1 - (rng.choice(neg, len(neg), replace=True) >= thr).mean()
        j_draws = draws_se + draws_sp - 1
        j_lo, j_hi = np.percentile(j_draws, [2.5, 97.5])

        row = {
            "detector": detector,
            "genre": genre,
            "sensitivity": se,
            "specificity": sp,
            "youden_j": se + sp - 1,
            "j_lo": j_lo,
            "j_hi": j_hi,
            "informative": bool(j_lo > 0),
        }
        for ap in AP_GRID:
            corrected = rogan_gladen(ap, draws_se, draws_sp)
            finite = corrected[np.isfinite(corrected)]
            key = f"ap{int(ap * 100):02d}"
            # Unclipped, because the interesting failure is the one that leaves the unit
            # interval. A detector with sensitivity below the rate it is asked to
            # explain implies a prevalence above one, which is not a large estimate but
            # a contradiction: no prevalence at all could have produced that many flags
            # through this instrument, so the flagged set contains something else.
            raw = (ap + sp - 1) / (se + sp - 1) if (se + sp - 1) > 0 else np.nan
            row[f"{key}_raw"] = float(raw)
            row[f"{key}_point"] = float(rogan_gladen(ap, se, sp))
            row[f"{key}_lo"] = float(np.percentile(finite, 2.5)) if len(finite) else np.nan
            row[f"{key}_hi"] = float(np.percentile(finite, 97.5)) if len(finite) else np.nan
            # Share of resamples in which the instrument carries no information at all.
            row[f"{key}_undefined"] = float(np.isnan(corrected).mean())
            row[f"{key}_status"] = (
                "undefined" if not np.isfinite(raw)
                else "out of range" if raw > 1
                else "estimable"
            )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["detector", "genre"])
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "rq3_correction.parquet", index=False)

    print(f"\nYouden's J at the {TARGET_FPR:.0%} target FPR, {args.bootstrap} bootstrap draws\n")
    show = out.copy()
    show["J [95% CI]"] = [
        f"{r.youden_j:+.3f} [{r.j_lo:+.3f}, {r.j_hi:+.3f}]" for r in out.itertuples()
    ]
    show["Se"] = (100 * show.sensitivity).round(1)
    show["Sp"] = (100 * show.specificity).round(1)
    print(show[["detector", "genre", "Se", "Sp", "J [95% CI]", "informative"]]
          .to_string(index=False))

    print(f"\ncorrected prevalence per cent, or why the correction fails "
          f"({int(out.informative.sum())} of {len(out)} cells have J bounded above zero)\n")
    table = out[["detector", "genre"]].copy()
    for ap in AP_GRID:
        key = f"ap{int(ap * 100):02d}"
        table[f"AP={ap:.0%}"] = [
            r[f"{key}_status"] if r[f"{key}_status"] != "estimable"
            else f"{100 * r[f'{key}_point']:.1f} [{100 * r[f'{key}_lo']:.1f}, "
                 f"{100 * r[f'{key}_hi']:.1f}]"
            for _, r in out.iterrows()
        ]
    print(table.to_string(index=False))

    counts = pd.Series(
        [out[f"ap{int(ap * 100):02d}_status"].value_counts() for ap in AP_GRID][0]
    )
    print(f"\nat AP={AP_GRID[0]:.0%}: " + ", ".join(f"{v} {k}" for k, v in counts.items()))

    print(f"\nwrote {args.out / 'rq3_correction.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
