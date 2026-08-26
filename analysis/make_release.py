#!/usr/bin/env python3
"""Bundle the artifacts that cannot be regenerated without a GPU.

The scoring runs cost about fourteen hours on an H100, and the host they ran on is
decommissioned. The score files are therefore the one part of this project that a
replicator cannot reproduce cheaply, and they must outlive any single machine. They are
also safe to publish: each row is an identifier, a detector, its pinned version and a
number, with no artifact text in any column.

What the bundle deliberately leaves out is the corpus text itself. Redistributing the
bodies of several thousand repositories exceeds what their licences allow, which is why
the corpus is released as a specification and rebuild code instead. The manifest that
pins the specification travels with the bundle so that the two can be matched up.

Usage:
    python analysis/make_release.py
    python analysis/make_release.py --out /tmp/release
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

# Explicit rather than glob: an accidental match here would publish repository text.
INCLUDE = [
    ("scores", PROCESSED / "scores"),
    ("scores_superseded", PROCESSED / "scores_superseded"),
    ("validity", PROCESSED / "validity"),
    ("correction", PROCESSED / "correction"),
    ("kg", PROCESSED / "kg"),
    ("corpus_manifest.json", PROCESSED / "corpus_v1" / "corpus_manifest.json"),
    ("aidev_profile.json", PROCESSED / "aidev_profile.json"),
    ("crossd_probe.json", PROCESSED / "crossd_probe.json"),
    ("aidev_download_manifest.json",
     ROOT / "analysis" / "01_corpus" / "manifests" / "aidev_68ed5f4b80d2.json"),
]

# Named so the exclusion is a decision on the record rather than an omission.
EXCLUDE_REASON = {
    "corpus_v1/corpus.parquet": "contains repository text; the same rows without the "
                                "text column are included as corpus_metadata.parquet",
    "corpus_v1/n1_*.parquet": "contains repository text",
    "llm_judge_cache/": "redundant with llm_judge.parquet, and 323 MB of tiny files",
}


def corpus_metadata(tmp: Path) -> Path:
    """The corpus with the text column dropped.

    Without this the bundle is not self-sufficient: a score file is a list of
    identifiers and numbers, and nothing in it says which set or genre a row belongs to,
    so none of the tables could be recomputed. The metadata carries the labels and
    strata and no artifact text, which is exactly the line the release draws.
    """
    import pandas as pd

    frame = pd.read_parquet(PROCESSED / "corpus_v1" / "corpus.parquet")
    if "text" not in frame.columns:
        raise SystemExit("corpus has no text column; refusing to guess what to drop")
    out = tmp / "corpus_metadata.parquet"
    frame.drop(columns=["text"]).to_parquet(out, index=False)
    return out


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "release")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp())
    files: list[tuple[str, Path]] = [("corpus_metadata.parquet", corpus_metadata(tmp))]
    for name, source in INCLUDE:
        if not source.exists():
            print(f"  missing, skipped: {source.relative_to(ROOT)}")
            continue
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_file() and path.name != ".DS_Store":
                    files.append((f"{name}/{path.relative_to(source)}", path))
        else:
            files.append((name, source))

    manifest = {
        "generated": datetime.now(UTC).isoformat(),
        "note": "Scoring artifacts for the AI-authorship validity audit. No artifact "
                "text is included; see excluded for what is deliberately absent.",
        "excluded": EXCLUDE_REASON,
        "files": [
            {"path": arc, "bytes": path.stat().st_size, "sha256": digest(path)}
            for arc, path in files
        ],
    }
    total = sum(f["bytes"] for f in manifest["files"])
    manifest["total_bytes"] = total

    manifest_path = args.out / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    archive = args.out / "scoring-artifacts-v1.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for arc, path in files:
            tar.add(path, arcname=f"scoring-artifacts-v1/{arc}")
        tar.add(manifest_path, arcname="scoring-artifacts-v1/MANIFEST.json")

    print(f"{len(files)} files, {total / 1e6:.1f} MB uncompressed")
    print(f"  {archive.relative_to(ROOT)}  ({archive.stat().st_size / 1e6:.1f} MB)")
    print(f"  {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
