#!/usr/bin/env python3
"""Supervised in-domain baseline, after Ghaleb (arXiv:2601.17406).

The zero-shot detectors are off-the-shelf instruments applied to a domain they were not
built for. This baseline answers the obvious next question: how much of the agent/human
distinction is available in these artifacts at all, to a classifier that gets to see the
training distribution? It is the ceiling the zero-shot numbers should be read against.

Ghaleb extracts 53 features from agent pull requests and reduces them to 41, spanning
commit message patterns, pull request structure, code changes, patch-level code
characteristics and temporal signals, then fits XGBoost with five-fold stratified
cross-validation and reports 97.2% weighted F1. Three deliberate departures:

  * **The task differs.** He separates five agents from each other on contributions
    already known to be agent-authored. We separate agent-authored from pre-LLM human
    text, which is the decision the prevalence literature actually makes.

  * **Temporal features are excluded, not unavailable.** Submission hour, weekend and
    day-of-week separate our two classes perfectly and for the wrong reason: the
    negatives predate November 2022 by construction. Including them would measure the
    corpus design rather than the writing. The same applies to the pull-request-level
    aggregates over commits, files touched and change concentration, which our corpus
    does not carry per text and which `run_detectors.py` deliberately keeps away from
    every other detector.

  * **Only text-derived features remain**, which is the point rather than a compromise:
    a ceiling that sees more than the detectors it bounds is not a ceiling. What is left
    is 24 features computed from the artifact text alone, listed in FEATURES below.

XGBoost is replaced by scikit-learn's histogram gradient boosting, the same model
family, so that the replication package installs from pip without an OpenMP toolchain.

Folds are grouped by repository wherever the corpus records one, so a classifier cannot
memorise a project's house style in training and be rewarded for it in test. Positives
carry a repository for roughly a quarter of rows; the rest form singleton groups, which
is the conservative reading. Exact duplicate texts are dropped before splitting.

Usage:
    python fingerprint.py
    python fingerprint.py --folds 5 --seed 7
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _registry import register  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus_v1" / "corpus.parquet"
SCORES = ROOT / "data" / "processed" / "scores"

VERSION = "fingerprint/hgb/f24/grouped-5fold/2026-08-13"

CONVENTIONAL = re.compile(
    r"^(feat|fix|chore|docs|refactor|test|perf|build|ci|style|revert)(\([^)]+\))?!?:\s"
)
URL = re.compile(r"https?://\S+")
CODE_FENCE = re.compile(r"```")
CHECKBOX = re.compile(r"^\s*[-*]\s+\[[ xX]\]", re.MULTILINE)
BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
HEADING = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
COMMENT_LINE = re.compile(r"^\s*(#|//|/\*|\*|--)", re.MULTILINE)
IMPORT_LINE = re.compile(r"^\s*(import|from|#include|using|require)\b", re.MULTILINE)
FUNC_DEF = re.compile(r"\b(def|function|func|fn)\b|=>")
CLASS_DEF = re.compile(r"\b(class|struct|interface|enum)\b")
CONDITIONAL = re.compile(r"\b(if|else|elif|switch|case|when)\b")
LOOP = re.compile(r"\b(for|while|foreach|loop)\b")

# The feature names, in the order build_features emits them. Kept explicit so the count
# reported in the paper cannot drift away from what the model was given.
FEATURES = [
    "n_chars", "n_words", "n_lines", "mean_line_len", "max_line_len", "std_line_len",
    "multiline", "blank_line_ratio", "trailing_ws_ratio", "indent_ratio",
    "mean_indent", "upper_ratio", "digit_ratio", "punct_ratio", "capitalised_start",
    "conventional_prefix", "n_urls", "n_code_fences", "n_checkboxes", "n_bullets",
    "n_headings", "comment_density", "import_density", "structure_density",
]


def build_features(texts: list[str]) -> np.ndarray:
    rows = []
    for raw in texts:
        t = raw or ""
        lines = t.split("\n")
        lengths = np.array([len(x) for x in lines], dtype=float)
        stripped = [x for x in lines if x.strip()]
        n_lines = max(len(lines), 1)
        n_chars = max(len(t), 1)
        indents = [len(x) - len(x.lstrip()) for x in stripped]
        rows.append([
            len(t),
            len(t.split()),
            len(lines),
            float(lengths.mean()) if len(lengths) else 0.0,
            float(lengths.max()) if len(lengths) else 0.0,
            float(lengths.std()) if len(lengths) > 1 else 0.0,
            1.0 if len(lines) > 1 else 0.0,
            sum(1 for x in lines if not x.strip()) / n_lines,
            sum(1 for x in lines if x != x.rstrip()) / n_lines,
            sum(1 for x in indents if x > 0) / max(len(stripped), 1),
            float(np.mean(indents)) if indents else 0.0,
            sum(1 for ch in t if ch.isupper()) / n_chars,
            sum(1 for ch in t if ch.isdigit()) / n_chars,
            sum(1 for ch in t if not ch.isalnum() and not ch.isspace()) / n_chars,
            1.0 if t[:1].isupper() else 0.0,
            1.0 if CONVENTIONAL.search(t) else 0.0,
            len(URL.findall(t)),
            len(CODE_FENCE.findall(t)),
            len(CHECKBOX.findall(t)),
            len(BULLET.findall(t)),
            len(HEADING.findall(t)),
            len(COMMENT_LINE.findall(t)) / n_lines,
            len(IMPORT_LINE.findall(t)) / n_lines,
            (len(FUNC_DEF.findall(t)) + len(CLASS_DEF.findall(t))
             + len(CONDITIONAL.findall(t)) + len(LOOP.findall(t))) / n_lines,
        ])
    return np.asarray(rows, dtype=float)


def out_of_fold_scores(frame: pd.DataFrame, n_rows: int, folds: int, seed: int) -> np.ndarray:
    """Agent-class probability per corpus row, from a model that never saw that row.

    Returned aligned to the full corpus and NaN outside the evaluated selection. The
    rest of the pipeline attaches scores to corpus rows by position rather than by
    source_id, which is not unique, so a score file that covers a subset still has to
    be corpus-length and in corpus order.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    filled = np.full(n_rows, np.nan)
    for genre, sub in frame.groupby("genre"):
        # One row per distinct text: a duplicate would otherwise be able to sit in the
        # training fold and the test fold at once. The score is copied back to its
        # duplicates afterwards.
        rep = sub.drop_duplicates(subset="text")
        y = (rep.set == "P").to_numpy(int)
        if y.sum() < 100 or (1 - y).sum() < 100:
            print(f"  {genre:15} skipped, {y.sum()} positives / {(1 - y).sum()} negatives")
            continue

        # Rows without a repository become singleton groups: never grouped with another
        # row, so they can never leak a project's style across the split.
        repo = rep.repo.fillna("").to_numpy()
        groups = np.where(repo == "", np.array([f"row:{p}" for p in rep.pos]), "repo:" + repo)

        X = build_features(rep.text.tolist())
        pred = np.full(len(rep), np.nan)
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for train, test in splitter.split(X, y, groups):
            model = HistGradientBoostingClassifier(
                max_depth=6, max_iter=100, random_state=seed
            )
            model.fit(X[train], y[train])
            pred[test] = model.predict_proba(X[test])[:, 1]

        by_text = dict(zip(rep.text, pred))
        filled[sub.pos.to_numpy()] = sub.text.map(by_text).to_numpy(float)
        auroc = _auroc(pred[y == 1], pred[y == 0])
        print(f"  {genre:15} {len(rep):>6,} texts, {y.sum():>5,} positive, AUROC {auroc:.3f}")
    return filled


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    from scipy.stats import rankdata

    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    args = ap.parse_args()

    corpus = pd.read_parquet(args.corpus).reset_index(drop=True)
    corpus["pos"] = np.arange(len(corpus))
    # Trained against the matched pre-LLM negatives, the same comparison RQ1 reports for
    # every other detector. The unmatched pool differs in length, and length is a
    # confound the matching exists to remove.
    frame = corpus[
        (corpus.set == "P") | ((corpus.set == "N1") & (corpus.matched == True))  # noqa: E712
    ]
    print(f"{len(frame):,} texts, {args.folds}-fold grouped cross-validation, "
          f"{len(FEATURES)} features\n")

    scores = out_of_fold_scores(frame, len(corpus), args.folds, args.seed)
    SCORES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "source_id": corpus.source_id,
            "detector": "fingerprint",
            "version": VERSION,
            "score": scores,
        }
    ).to_parquet(SCORES / "fingerprint.parquet", index=False)

    (SCORES / "fingerprint_manifest.json").write_text(
        json.dumps(
            {
                "generated": datetime.now(UTC).isoformat(),
                "version": VERSION,
                "features": FEATURES,
                "folds": args.folds,
                "seed": args.seed,
                "n_texts": int(np.isfinite(scores).sum()),
            },
            indent=2,
        )
    )
    print(f"\nwrote {SCORES / 'fingerprint.parquet'}")
    return 0


@register("fingerprint", kind="supervised", needs_gpu=False,
          note="cross-validated; produced by running fingerprint.py, not by the sweep")
def _build():
    raise RuntimeError(
        "fingerprint is fit by cross-validation and cannot score arbitrary text. "
        "Run: python analysis/02_detectors/fingerprint.py"
    )


if __name__ == "__main__":
    sys.exit(main())
