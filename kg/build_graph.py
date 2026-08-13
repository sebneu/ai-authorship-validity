#!/usr/bin/env python3
"""RQ4: turn the scored corpus into authorship claims with their evidence.

Every artifact in the corpus is emitted as an aiprov:Artifact. Each way of knowing who
wrote it becomes a separate aiprov:AuthorshipClaim carrying its own evidence:

  * the AIDev positives carry declared evidence, an agent account or a trailer;
  * every detector that flagged an artifact at its operating point contributes a
    detector-evidence claim, annotated with the false positive rate, sensitivity and
    Youden's J measured for that detector, genre and threshold in RQ1 and RQ3.

Keeping declaration and inference as sibling claims about one artifact is the whole
design. It lets a consumer ask which attributions rest only on inference, and how well
that inference performs on the stratum in question, without re-running anything.

The graph is built over a sample by default. Its purpose is to demonstrate the model and
answer the competency questions, not to serve as a second dataset.

Usage:
    python build_graph.py
    python build_graph.py --sample 8000 --out ../data/processed/kg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdflib import XSD, Graph, Literal, Namespace, URIRef
from rdflib.namespace import PROV, RDF, RDFS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "03_validity"))

from metrics import load  # noqa: E402

AIPROV = Namespace("https://w3id.org/aiprov#")
EX = Namespace("https://w3id.org/aiprov/corpus/")

METRICS = ROOT / "data" / "processed" / "validity" / "rq1_metrics.parquet"
CORRECTION = ROOT / "data" / "processed" / "correction" / "rq3_correction.parquet"


def build(df: pd.DataFrame, stratum: pd.DataFrame, sample: int, seed: int) -> Graph:
    g = Graph()
    g.bind("aiprov", AIPROV)
    g.bind("prov", PROV)
    g.bind("ex", EX)

    # Sample artifacts, then keep every detector row belonging to them, so that each
    # artifact in the graph carries its full evidence rather than a random subset of it.
    artifacts = df[["source_id", "genre", "set", "agent", "repo", "n_chars"]].drop_duplicates(
        subset=["source_id", "genre"]
    )
    if sample and sample < len(artifacts):
        artifacts = artifacts.sample(sample, random_state=seed)
    keep = set(zip(artifacts.source_id, artifacts.genre))
    rows = df[[k in keep for k in zip(df.source_id, df.genre)]]

    by_stratum = stratum.set_index(["detector", "genre"])

    # One activity node per detector run, shared by every piece of evidence it produced.
    for (detector, version), _ in rows.groupby(["detector", "version"]):
        act = EX[f"run/{detector}"]
        g.add((act, RDF.type, AIPROV.DetectionActivity))
        g.add((act, AIPROV.detector, Literal(detector)))
        g.add((act, AIPROV.detectorVersion, Literal(version)))

    for row in artifacts.itertuples():
        art = EX[f"artifact/{row.genre}/{abs(hash(row.source_id)) % (10 ** 12)}"]
        g.add((art, RDF.type, AIPROV.Artifact))
        g.add((art, AIPROV.genre, Literal(row.genre)))
        g.add((art, AIPROV.characterLength, Literal(int(row.n_chars), datatype=XSD.integer)))
        if isinstance(row.repo, str) and row.repo:
            g.add((art, AIPROV.repository, Literal(row.repo)))

        # Declared evidence, present only for the agent-attributed positives. The
        # pre-language-model negatives get no claim at all: absence of a claim is the
        # correct representation of "nobody said anything", and is not the same as a
        # claim of human authorship.
        if row.set == "P" and isinstance(row.agent, str):
            agent = EX[f"agent/{row.agent}"]
            g.add((agent, RDF.type, AIPROV.CodingAgent))
            g.add((agent, AIPROV.tool, Literal(row.agent)))
            claim = EX[f"claim/declared/{row.genre}/{abs(hash(row.source_id)) % (10 ** 12)}"]
            ev = EX[f"evidence/declared/{row.genre}/{abs(hash(row.source_id)) % (10 ** 12)}"]
            g.add((claim, RDF.type, AIPROV.AuthorshipClaim))
            g.add((claim, AIPROV.claimAbout, art))
            g.add((claim, AIPROV.claimsAgent, agent))
            g.add((claim, AIPROV.hasEvidence, ev))
            g.add((ev, RDF.type, AIPROV.AgentAccountEvidence))
            g.add((art, PROV.wasAttributedTo, agent))

    # Detector evidence: one claim per detector that flagged the artifact.
    for row in rows.itertuples():
        key = (row.detector, row.genre)
        if key not in by_stratum.index or not np.isfinite(row.score):
            continue
        s = by_stratum.loc[key]
        if row.score < s.threshold:
            continue
        ident = f"{row.detector}/{row.genre}/{abs(hash(row.source_id)) % (10 ** 12)}"
        art = EX[f"artifact/{row.genre}/{abs(hash(row.source_id)) % (10 ** 12)}"]
        claim = EX[f"claim/detected/{ident}"]
        ev = EX[f"evidence/detected/{ident}"]
        g.add((claim, RDF.type, AIPROV.AuthorshipClaim))
        g.add((claim, AIPROV.claimAbout, art))
        g.add((claim, AIPROV.hasEvidence, ev))
        g.add((ev, RDF.type, AIPROV.DetectorEvidence))
        g.add((ev, AIPROV.producedBy, EX[f"run/{row.detector}"]))
        g.add((ev, AIPROV.score, Literal(float(row.score), datatype=XSD.double)))
        g.add((ev, AIPROV.threshold, Literal(float(s.threshold), datatype=XSD.double)))
        g.add((ev, AIPROV.stratumFalsePositiveRate,
               Literal(float(s.fpr_achieved), datatype=XSD.double)))
        g.add((ev, AIPROV.stratumSensitivity,
               Literal(float(s.recall_at_fpr), datatype=XSD.double)))
        if np.isfinite(s.youden_j):
            g.add((ev, AIPROV.stratumYoudenJ, Literal(float(s.youden_j), datatype=XSD.double)))
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # 2,500 artifacts keeps the competency queries answerable in seconds. rdflib
    # evaluates FILTER NOT EXISTS per solution, so the two queries that ask what an
    # artifact is *not* corroborated by grow steeply with graph size.
    ap.add_argument("--sample", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "kg")
    args = ap.parse_args()

    print("loading scores:")
    df = load()
    metrics = pd.read_parquet(METRICS)[
        ["detector", "genre", "threshold", "fpr_achieved", "recall_at_fpr"]
    ]
    if CORRECTION.exists():
        j = pd.read_parquet(CORRECTION)[["detector", "genre", "youden_j"]]
        metrics = metrics.merge(j, on=["detector", "genre"], how="left")
    else:
        metrics["youden_j"] = np.nan

    graph = build(df, metrics, args.sample, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "aiprov_corpus.ttl"
    graph.serialize(target, format="turtle")

    claims = len(set(graph.subjects(RDF.type, AIPROV.AuthorshipClaim)))
    detected = len(set(graph.subjects(RDF.type, AIPROV.DetectorEvidence)))
    declared = len(set(graph.subjects(RDF.type, AIPROV.AgentAccountEvidence)))
    print(f"\n{len(graph):,} triples, {claims:,} authorship claims "
          f"({declared:,} declared, {detected:,} detector-only pieces of evidence)")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
