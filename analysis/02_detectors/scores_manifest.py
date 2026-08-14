#!/usr/bin/env python3
"""Rebuild the score manifest from the score files themselves.

Provenance has to survive the way this pipeline actually runs. The GPU detectors were
scored on a separate host across several sessions, and early runs used a driver that
rewrote the manifest per run instead of merging, so the recorded history of the first
runs was overwritten by the last. The version strings survived, because each adapter
writes its own into every row it produces.

This reads the parquet files and reconstructs the manifest from what is in them, then
folds in any timing recorded by a run that did merge properly. The result is a manifest
that cannot disagree with the data, since it is derived from it.

Usage:
    python scores_manifest.py
    python scores_manifest.py --merge /path/to/host/run_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "data" / "processed" / "scores"
CORPUS = ROOT / "data" / "processed" / "corpus_v1" / "corpus.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merge", type=Path, action="append", default=[],
                    help="another run_manifest.json to fold timing in from")
    args = ap.parse_args()

    corpus = pd.read_parquet(CORPUS, columns=["source_id", "genre"]).reset_index(drop=True)
    manifest_path = SCORES / "run_manifest.json"

    existing: dict[str, dict] = {}
    for source in [manifest_path, *args.merge]:
        if source and source.exists():
            existing.update(json.loads(source.read_text()).get("runs", {}))

    runs: dict[str, dict] = {}
    for path in sorted(SCORES.glob("*.parquet")):
        scores = pd.read_parquet(path).reset_index(drop=True)
        name = path.stem
        scored = scores.score.notna()
        # Which genres this file covers, recovered positionally: a genre-restricted run
        # writes rows in corpus order for that genre alone.
        if len(scores) == len(corpus):
            genres = sorted(corpus.loc[scored.to_numpy(), "genre"].unique().tolist())
        else:
            matched = [
                g for g in sorted(corpus.genre.unique())
                if (corpus.genre == g).sum() == len(scores)
                and (corpus.loc[corpus.genre == g, "source_id"].to_numpy()
                     == scores.source_id.to_numpy()).all()
            ]
            genres = matched or ["unaligned"]

        prior = existing.get(name, {})
        runs[name] = {
            "detector": name,
            "version": sorted(scores.version.dropna().unique().tolist()),
            "rows": int(len(scores)),
            "scored": int(scored.sum()),
            "genres": genres,
            "status": "ok" if scored.any() else "empty",
            # Timing only exists where a merging driver recorded it; absence is recorded
            # rather than guessed.
            "seconds": prior.get("seconds"),
            "generated": prior.get("generated"),
        }

    manifest_path.write_text(json.dumps(
        {
            "rebuilt": datetime.now(UTC).isoformat(),
            "corpus": str(CORPUS.relative_to(ROOT)),
            "corpus_rows": int(len(corpus)),
            "runs": runs,
        },
        indent=2, sort_keys=True,
    ))

    print(f"{len(runs)} detector(s) in {manifest_path.relative_to(ROOT)}\n")
    for name, run in sorted(runs.items()):
        timing = f"{run['seconds']:.0f}s" if run["seconds"] else "timing not recorded"
        print(f"  {name:16} {run['scored']:>6,}/{run['rows']:,} scored  "
              f"{'/'.join(run['genres'])}  {timing}")
        for version in run["version"]:
            print(f"  {'':16} {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
