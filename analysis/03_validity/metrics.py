#!/usr/bin/env python3
"""RQ1 and RQ2 metrics: discrimination and false positive rates per stratum.

Joins the frozen corpus to every detector's scores and reports, per detector and genre:

  AUROC against the matched pre-LLM negatives (N1). Matched is the honest comparison --
  the unmatched sets differ in length, and length is a confound rather than a signal.

  AUROC against contemporaneous human pull requests (N2), which removes the temporal
  confound at the cost of certainty about the negatives.

  False positive rate on N1 and on N3 at each detector's operating point. On N1 every
  detection is an error by construction; N3 shows how much of it is decade-old
  automation rather than model output.

Confidence intervals are bootstrap percentile intervals over texts. With no calibrated
threshold published for most of these detectors, the operating point is taken as the
score at a fixed quantile of the negative distribution and reported as such.

Usage:
    python metrics.py
    python metrics.py --bootstrap 2000 --out ../../data/processed/validity
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus_v1" / "corpus.parquet"
SCORES = ROOT / "data" / "processed" / "scores"

# Operating point: the threshold that would flag this share of pre-LLM text. Detectors
# in this literature are applied without in-domain calibration, so we fix a plausible
# target FPR and report what recall it actually buys.
TARGET_FPR = 0.05


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC, ties averaged. NaN when either side is empty."""
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_ci(pos: np.ndarray, neg: np.ndarray, n: int, rng: np.random.Generator) -> tuple:
    if n <= 0 or len(pos) == 0 or len(neg) == 0:
        return (float("nan"), float("nan"))
    vals = [
        auroc(rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True))
        for _ in range(n)
    ]
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def rate_at(values: np.ndarray, threshold: float) -> float:
    """Share of values at or above the threshold. Inclusive because these detectors
    produce heavily tied scores, where a strict comparison can flag nothing at all."""
    clean = values[~np.isnan(values)]
    if len(clean) == 0 or np.isnan(threshold):
        return float("nan")
    return float((clean >= threshold).mean())


def operating_point(pos: np.ndarray, neg: np.ndarray, target: float) -> tuple:
    """Threshold whose false positive rate comes closest to the target without
    exceeding it, plus the FPR actually achieved and the recall there.

    Taking a quantile of the negatives fails on tied scores: the LLM judge puts 57% of
    its answers on exactly 0.0 and 25% on exactly 1.0, so the 95th percentile lands on
    the maximum and a strict threshold flags nothing. Walking the distinct score values
    reports what the detector can actually deliver, and the achieved FPR alongside it
    makes the coarseness visible instead of hiding it.
    """
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return (float("nan"),) * 3
    thresholds = np.unique(np.concatenate([pos, neg]))[::-1]
    # Most conservative available point, used when the target is unreachable. The LLM
    # judge puts 5.9% of pre-LLM pull request bodies on "certainly AI", so no threshold
    # can hold its false positive rate under 5% -- reporting the achievable 5.9% and
    # the recall there is more use than reporting nothing.
    top = float(thresholds[0])
    best = (top, float((neg >= top).mean()), float((pos >= top).mean()))
    for thr in thresholds:
        fpr = float((neg >= thr).mean())
        if fpr > target:
            break
        best = (float(thr), fpr, float((pos >= thr).mean()))
    return best


def align(corpus: pd.DataFrame, scores: pd.DataFrame, name: str) -> pd.DataFrame | None:
    """Find the corpus rows a score file corresponds to.

    Detectors may be run on a subset: DetectCodeGPT costs 51 forward passes per text,
    so it runs on diffs alone. The written file then holds fewer rows than the corpus
    and in the order of the filtered frame. We recover the subset by testing the full
    corpus first, then each single genre, comparing the source_id sequence rather than
    trusting length alone.
    """
    if len(scores) == len(corpus) and (scores.source_id.values == corpus.source_id.values).all():
        return corpus
    for genre in sorted(corpus.genre.dropna().unique()):
        subset = corpus[corpus.genre == genre].reset_index(drop=True)
        if len(subset) == len(scores) and (
            subset.source_id.values == scores.source_id.values
        ).all():
            return subset
    return None


def load() -> pd.DataFrame:
    """Attach scores to corpus rows positionally, not by source_id.

    source_id is not unique: a matched negative is a second copy of an unmatched one,
    and a few artifacts appear in more than one cell. Joining on it fans out, which
    inflated an earlier version of this table by 794 rows. run_detectors writes one
    score per corpus row in corpus order, so position is the correct key -- and the
    assertion below fails loudly if that ever stops holding.
    """
    if not CORPUS.exists():
        raise SystemExit(f"missing {CORPUS}")
    corpus = pd.read_parquet(
        CORPUS, columns=["source_id", "set", "genre", "agent", "n_chars", "matched"]
    ).reset_index(drop=True)
    files = sorted(SCORES.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no score files in {SCORES}")

    frames = []
    for path in files:
        scores = pd.read_parquet(path).reset_index(drop=True)
        rows = align(corpus, scores, path.name)
        if rows is None:
            # Never continue past this. An unalignable score file means the corpus was
            # rebuilt under a detector, and every number downstream would be computed
            # against the wrong texts while looking entirely healthy.
            raise SystemExit(
                f"{path.name}: {len(scores):,} scores match neither the corpus "
                f"({len(corpus):,} rows) nor any single genre. Re-run this detector "
                f"against the current corpus."
            )
        merged = rows.copy()
        for col in ("detector", "version", "score"):
            merged[col] = scores[col].values
        scored = merged.score.notna()
        genres = "/".join(sorted(merged.loc[scored, "genre"].unique()))
        note = "" if scored.all() else f", {(~scored).sum():,} unscored"
        print(f"  {path.stem:18} {scored.sum():>7,} scored, "
              f"{merged.score.nunique():>6,} distinct{note}, {genres}")
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Scored rows per detector and genre, with the share that came back missing.

    A detector that silently drops a genre, or returns NaN for part of one, still
    produces a full table of plausible-looking metrics. This makes the gaps explicit
    before any of them are read.
    """
    rows = []
    for (detector, genre), sub in df.groupby(["detector", "genre"]):
        rows.append(
            {
                "detector": detector,
                "genre": genre,
                "n": len(sub),
                "scored": int(sub.score.notna().sum()),
                "missing_pct": round(100 * sub.score.isna().mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "validity")
    args = ap.parse_args()

    print("loading scores:")
    df = load()
    rng = np.random.default_rng(7)

    gaps = coverage(df)
    incomplete = gaps[(gaps.missing_pct > 0) & (gaps.missing_pct < 100)]
    if len(incomplete):
        print("\npartially scored cells:")
        print(incomplete.to_string(index=False))

    rows = []
    for detector, dsub in df.groupby("detector"):
        for genre, gsub in dsub.groupby("genre"):
            pos = gsub.loc[gsub.set == "P", "score"].to_numpy(float)
            # Matched N1 only: the unmatched pool differs in length distribution.
            n1 = gsub.loc[(gsub.set == "N1") & (gsub.matched == True), "score"].to_numpy(float)  # noqa: E712
            n1_all = gsub.loc[gsub.set == "N1", "score"].to_numpy(float)
            n2 = gsub.loc[gsub.set == "N2", "score"].to_numpy(float)
            n3 = gsub.loc[gsub.set == "N3", "score"].to_numpy(float)
            if len(pos) == 0 or len(n1) == 0:
                continue

            a1 = auroc(pos, n1)
            lo, hi = boot_ci(pos, n1, args.bootstrap, rng)
            thr, achieved, recall = operating_point(pos, n1, TARGET_FPR)
            rows.append(
                {
                    "detector": detector,
                    "genre": genre,
                    "n_pos": len(pos),
                    "n_neg": len(n1),
                    "distinct_scores": int(np.unique(pos[~np.isnan(pos)]).size),
                    "auroc_n1": a1,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "auroc_n2": auroc(pos, n2) if len(n2) else float("nan"),
                    "threshold": thr,
                    "fpr_achieved": achieved,
                    "recall_at_fpr": recall,
                    "fpr_n1_unmatched": rate_at(n1_all, thr),
                    "fpr_n3_bots": rate_at(n3, thr) if len(n3) else float("nan"),
                }
            )

    out = pd.DataFrame(rows).sort_values(["detector", "genre"])
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out / "rq1_metrics.parquet", index=False)

    print(f"\nAUROC against matched pre-LLM negatives (N1), {args.bootstrap} bootstrap draws")
    print(f"recall reported at the threshold giving {TARGET_FPR:.0%} FPR on N1\n")
    show = out.copy()
    show["AUROC [95% CI]"] = [
        f"{r.auroc_n1:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" for r in out.itertuples()
    ]
    show["recall"] = (100 * show.recall_at_fpr).round(1)
    show["FPR ach."] = (100 * show.fpr_achieved).round(1)
    show["FPR bots"] = (100 * show.fpr_n3_bots).round(1)
    show["AUROC N2"] = show.auroc_n2.round(3)
    print(
        show[["detector", "genre", "n_pos", "AUROC [95% CI]", "FPR ach.", "recall",
              "FPR bots", "AUROC N2"]].to_string(index=False)
    )

    (args.out / "rq1_manifest.json").write_text(
        json.dumps(
            {
                "generated": datetime.now(UTC).isoformat(),
                "bootstrap": args.bootstrap,
                "target_fpr": TARGET_FPR,
                "detectors": sorted(df.detector.unique().tolist()),
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out / 'rq1_metrics.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
