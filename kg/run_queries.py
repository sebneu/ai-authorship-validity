#!/usr/bin/env python3
"""Run the competency questions against the provenance graph.

Each query in `queries/` states one question the model exists to answer. They are the
demonstration that the vocabulary earns its place: none of them is answerable from a
corpus of detector scores without the claim-and-evidence structure, because each turns
on the relation between an attribution and the grounds it rests on.

Usage:
    python run_queries.py
    python run_queries.py --graph ../data/processed/kg/aiprov_corpus.ttl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdflib import Graph

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = ROOT / "data" / "processed" / "kg" / "aiprov_corpus.ttl"


def shorten(value) -> str:
    text = str(value)
    if text.startswith("https://w3id.org/aiprov#"):
        return "aiprov:" + text.split("#", 1)[1]
    if text.startswith("http://www.w3.org/ns/prov#"):
        return "prov:" + text.split("#", 1)[1]
    try:
        return f"{float(text):.4g}"
    except (TypeError, ValueError):
        return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--queries", type=Path, default=Path(__file__).resolve().parent / "queries")
    ap.add_argument("--limit", type=int, default=12, help="rows shown per query")
    args = ap.parse_args()

    if not args.graph.exists():
        raise SystemExit(f"missing {args.graph}\nrun: python kg/build_graph.py")

    graph = Graph()
    graph.parse(args.graph, format="turtle")
    print(f"{len(graph):,} triples from {args.graph.name}")

    failures = 0
    for path in sorted(args.queries.glob("*.rq")):
        text = path.read_text()
        comment = "\n".join(
            line.lstrip("# ").rstrip() for line in text.splitlines() if line.startswith("#")
        )
        print(f"\n=== {path.stem} ===\n{comment}\n")
        rows = list(graph.query(text))
        if not rows:
            print("  no results")
            failures += 1
            continue
        header = [str(v) for v in rows[0].labels]
        widths = [max(len(h), 14) for h in header]
        print("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
        for row in rows[: args.limit]:
            print("  " + "  ".join(shorten(v).ljust(w) for v, w in zip(row, widths)))
        if len(rows) > args.limit:
            print(f"  ... {len(rows) - args.limit} more rows")

    if failures:
        # An empty competency question means the model does not in fact support the
        # question it was designed around, which is a modelling failure rather than a
        # quiet nil result.
        print(f"\n{failures} query/queries returned nothing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
