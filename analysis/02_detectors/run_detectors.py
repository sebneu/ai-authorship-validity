#!/usr/bin/env python3
"""Score the frozen corpus with each available detector.

Writes one parquet per detector, so a failure costs one detector rather than the sweep,
and so the GPU host and a laptop can each contribute without writing the same file.

Usage:
    python run_detectors.py --dry-run                  # what would run here
    python run_detectors.py --detectors heuristics
    python run_detectors.py --detectors all --limit 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _registry import available, get, load_adapters  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus_v1" / "corpus.parquet"
SCORES = ROOT / "data" / "processed" / "scores"

# Columns a detector must never see. Every one of them correlates with the label, so a
# detector given any of them could score well without reading the text.
FORBIDDEN = {"set", "label", "agent", "repo", "created_at", "matched", "ext", "genre"}


def relative_if_possible(path: Path) -> str:
    """Path relative to the repo root when it is inside it, absolute otherwise.

    The corpus may be given as a relative path or live outside the tree on the GPU
    host, and the manifest should record it either way rather than failing.
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def score_detector(name: str, texts: list[str], batch_size: int) -> tuple[list[float], str, float]:
    detector = get(name).factory()
    started = time.perf_counter()
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        scores.extend(detector.score(texts[i : i + batch_size]))
    elapsed = time.perf_counter() - started
    if len(scores) != len(texts):
        raise RuntimeError(f"{name} returned {len(scores)} scores for {len(texts)} texts")
    return scores, detector.version, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detectors", nargs="*", default=["all"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, help="score only the first N texts (smoke test)")
    ap.add_argument("--genres", nargs="*",
                    help="restrict to these genres (DetectCodeGPT is code-specific, and "
                         "at 51 passes per text the full corpus would take 30 hours)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    args = ap.parse_args()

    load_adapters()
    have = available()

    if args.dry_run:
        print(f"{len(have)} adapter(s) registered here:\n")
        for name, reg in sorted(have.items()):
            state = reg.unavailable or ("GPU required" if reg.needs_gpu else "ready")
            print(f"  {name:18} {reg.kind:11} {state}")
            if reg.note:
                print(f"  {'':18} {reg.note}")
        return 0

    if not args.corpus.exists():
        raise SystemExit(f"missing {args.corpus}\nrun: python ../01_corpus/freeze_corpus.py")

    corpus = pd.read_parquet(args.corpus)
    if args.genres:
        corpus = corpus[corpus.genre.isin(args.genres)]
        print(f"restricted to {args.genres}: {len(corpus):,} texts")
    if args.limit:
        corpus = corpus.head(args.limit)

    # The guard that matters: hand the adapters strings, never the frame.
    texts = corpus.text.fillna("").tolist()
    ids = corpus.source_id.tolist()
    leaked = FORBIDDEN & set(dir(texts))
    assert not leaked, f"text list carries corpus columns: {leaked}"

    wanted = sorted(have) if args.detectors == ["all"] else args.detectors
    unknown = [d for d in wanted if d not in have]
    if unknown:
        raise SystemExit(f"unknown detector(s): {unknown}\navailable: {sorted(have)}")

    SCORES.mkdir(parents=True, exist_ok=True)
    print(f"{len(texts):,} texts, {len(wanted)} detector(s)\n")

    summary = []
    for name in wanted:
        reg = have[name]
        if reg.unavailable:
            print(f"  {name}: skipped ({reg.unavailable})")
            continue
        try:
            scores, version, elapsed = score_detector(name, texts, args.batch_size)
        except Exception as exc:  # noqa: BLE001 - one detector must not stop the sweep
            print(f"  {name}: FAILED {str(exc)[:90]}")
            summary.append({"detector": name, "status": "failed", "error": str(exc)[:200]})
            continue

        out = SCORES / f"{name}.parquet"
        pd.DataFrame(
            {
                "source_id": ids,
                "detector": name,
                "version": version,
                "score": scores,
            }
        ).to_parquet(out, index=False)
        rate = len(texts) / max(elapsed, 1e-9)
        print(f"  {name}: {len(scores):,} scores in {elapsed:.1f}s ({rate:,.0f}/s) -> {out.name}")
        summary.append(
            {
                "detector": name,
                "status": "ok",
                "version": version,
                "n": len(scores),
                "seconds": round(elapsed, 2),
            }
        )

    # Merge rather than overwrite. Detectors are run in separate sessions -- the GPU
    # ones on the H100 host, the CPU ones here, DetectCodeGPT on its own because it
    # takes a night -- so a manifest rewritten per run describes only the last detector
    # and silently drops the provenance of every score file beside it.
    manifest_path = SCORES / "run_manifest.json"
    manifest = {"runs": {}}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        manifest["runs"] = previous.get("runs", {})
    for entry in summary:
        manifest["runs"][entry["detector"]] = {
            **entry,
            "generated": datetime.now(UTC).isoformat(),
            "corpus": relative_if_possible(args.corpus),
            "n_texts": len(texts),
            "genres": args.genres or "all",
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nwrote {SCORES}")
    return 0 if all(s["status"] == "ok" for s in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
