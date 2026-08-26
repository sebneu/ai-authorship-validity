#!/usr/bin/env python3
"""Unpack the release bundle into the layout the analysis expects.

A fresh clone has the code and the bundle but no `data/` payloads, because those are
gitignored. This puts the bundle back where the pipeline looks for it, so that a machine
with neither the original corpus nor a GPU can still regenerate every table and figure.

One substitution matters. The bundle carries `corpus_metadata.parquet`, which is the
corpus with the text column dropped, and the analysis reads `corpus.parquet`. Nothing
downstream of the score files touches the text -- the metrics read set, genre, agent,
repository, extension, length and matching status -- so the metadata stands in for the
corpus and is written under that name. The one stage that genuinely needs the text is
`fingerprint`, which computes features from it; its scores are already in the bundle, so
the stage can be skipped rather than re-run.

Usage:
    python analysis/restore_release.py
    python analysis/restore_release.py --bundle release/scoring-artifacts-v1.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# bundle path -> destination under data/processed/
PLACEMENT = {
    "scores": "scores",
    "scores_superseded": "scores_superseded",
    "validity": "validity",
    "correction": "correction",
    "kg": "kg",
    "corpus_manifest.json": "corpus_v1/corpus_manifest.json",
    "corpus_metadata.parquet": "corpus_v1/corpus.parquet",
    "aidev_profile.json": "aidev_profile.json",
    "crossd_probe.json": "crossd_probe.json",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path,
                    default=ROOT / "release" / "scoring-artifacts-v1.tar.gz")
    ap.add_argument("--force", action="store_true",
                    help="overwrite files that are already present")
    args = ap.parse_args()

    if not args.bundle.exists():
        raise SystemExit(f"missing {args.bundle}")

    tmp = Path(tempfile.mkdtemp())
    with tarfile.open(args.bundle) as tar:
        tar.extractall(tmp)
    root = tmp / "scoring-artifacts-v1"

    manifest = json.loads((root / "MANIFEST.json").read_text())
    expected = {f["path"]: f["sha256"] for f in manifest["files"]}

    written, skipped, bad = 0, 0, []
    for name, target in PLACEMENT.items():
        source = root / name
        if not source.exists():
            continue
        for path in ([source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())):
            arc = str(path.relative_to(root))
            dest = PROCESSED / (target if source.is_file()
                                else f"{target}/{path.relative_to(source)}")
            if arc in expected:
                got = hashlib.sha256(path.read_bytes()).hexdigest()
                if got != expected[arc]:
                    bad.append(arc)
                    continue
            if dest.exists() and not args.force:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            written += 1

    shutil.rmtree(tmp)
    if bad:
        # A checksum mismatch means the bundle is damaged, and silently restoring it
        # would put corrupt inputs under a pipeline that cannot tell.
        raise SystemExit(f"checksum mismatch, refusing to restore: {bad}")

    print(f"restored {written} file(s) into {PROCESSED.relative_to(ROOT)}"
          + (f", {skipped} already present (use --force to overwrite)" if skipped else ""))
    print("\ncorpus.parquet here is the text-free metadata, which is enough for:")
    print("  python analysis/run_all.py --skip fingerprint")
    print("The fingerprint stage needs the artifact text; its scores are already restored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
