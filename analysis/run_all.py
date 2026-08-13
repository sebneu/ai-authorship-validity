#!/usr/bin/env python3
"""Regenerate every result the manuscript quotes, in dependency order.

The corpus build and the GPU scoring runs are not included: they are long, they need
credentials or a GPU, and neither changes unless a deliberate decision changes it.
Everything downstream of the score files runs here, so after a detector lands its
parquet this is the single command that moves the paper.

    python analysis/run_all.py                  # everything below the score files
    python analysis/run_all.py --skip kg        # leave the graph alone
    python analysis/run_all.py --bootstrap 2000 # publication-grade intervals

The manuscript itself is built from paper/ with make, which re-runs check_manuscript.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# name -> (script, extra args). Order is the dependency order: RQ1 writes the metrics
# RQ3 reads, and the graph reads both.
STAGES = [
    ("corpus_table", ROOT / "analysis" / "01_corpus" / "corpus_table.py", []),
    ("fingerprint", ROOT / "analysis" / "02_detectors" / "fingerprint.py", []),
    ("rq1", ROOT / "analysis" / "03_validity" / "metrics.py", ["--bootstrap"]),
    ("rq2", ROOT / "analysis" / "03_validity" / "stratified.py", ["--bootstrap"]),
    ("published", ROOT / "analysis" / "03_validity" / "operating_points.py", []),
    ("rq3", ROOT / "analysis" / "04_correction" / "rogan_gladen.py", ["--bootstrap"]),
    ("tables", ROOT / "analysis" / "03_validity" / "tables.py", []),
    ("figures", ROOT / "analysis" / "03_validity" / "figures.py", []),
    ("kg", ROOT / "kg" / "build_graph.py", []),
    ("queries", ROOT / "kg" / "run_queries.py", []),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--skip", nargs="*", default=[], help="stage names to leave out")
    ap.add_argument("--only", nargs="*", help="run only these stages")
    args = ap.parse_args()

    failed = []
    for name, script, extra in STAGES:
        if name in args.skip or (args.only and name not in args.only):
            print(f"--- {name}: skipped")
            continue
        cmd = [PYTHON, str(script)]
        if extra == ["--bootstrap"]:
            cmd += ["--bootstrap", str(args.bootstrap)]
        print(f"\n=== {name}: {' '.join(cmd[1:])}")
        result = subprocess.run(cmd, cwd=script.parent)
        if result.returncode != 0:
            # Keep going: a later stage may still be reproducible, and stopping at the
            # first failure hides how much of the pipeline is actually broken.
            failed.append(name)
            print(f"--- {name}: FAILED with {result.returncode}")

    if failed:
        print(f"\n{len(failed)} stage(s) failed: {', '.join(failed)}")
        return 1
    print("\nall stages ok; build the manuscript with: make -C paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
