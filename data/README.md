# Data

Payloads in this directory are **gitignored**. This file is the provenance record: every
dataset `analysis/` reads must be listed here with enough detail to reconstruct it.

## Raw inputs

| Path | Source | Version / accessed | Role |
|---|---|---|---|
| `raw/aidev/` | AIDev \[[arXiv:2507.15003](https://arxiv.org/abs/2507.15003)], HF `hao-li/AIDev`, Zenodo [10.5281/zenodo.16919272](https://doi.org/10.5281/zenodo.16919272) | revision `68ed5f4b80d27a9e057fc57567f38bd322ac73ec`, hub last modified 2026-05-10, downloaded 2026-08-03 | Positive ground truth: pull requests opened by coding agents, plus the contemporaneous human control (N2) |

The dataset is live and has roughly doubled since the paper introducing it, so the
revision is pinned rather than resolved at run time. Per-file sizes and SHA-256 sums are
in `analysis/01_corpus/manifests/aidev_68ed5f4b80d2.json`.

**There is no `raw/github/` or `raw/crossd/`.** Both appeared in an earlier version of
this file and never existed:

- GitHub-sourced negatives are written straight into `processed/corpus_v1/` by the
  collectors in `analysis/01_corpus/`, because the raw API responses are large, are not
  redistributable, and are not needed once the fields we keep have been extracted. The
  collectors record their own selection predicates, and re-running them reconstructs the
  same rows from the API and from cloned repository history.
- The CrOSSD panel was evaluated as a source and rejected. It stores README text, issue
  bodies and issue comments but records no author, login or actor field on any of them,
  so it supports no human/agent/bot separation and no join to AIDev. The probe that
  established this is `processed/crossd_probe.json` (2026-08-03), and the finding is
  reported in the paper. No CrOSSD data enters any result.

## Processed

| Path | Produced by | Contents |
|---|---|---|
| `processed/corpus_v1/` | `analysis/01_corpus/freeze_corpus.py` | The frozen corpus (86,484 texts) plus the per-set intermediates and `corpus_manifest.json` (seed, per-genre cap, matching deficits, per-cell counts) |
| `processed/scores/` | `analysis/02_detectors/run_detectors.py`, `fingerprint.py` | One parquet per detector, each row carrying the adapter version and the model revision it loaded; `run_manifest.json` is rebuilt from those files by `scores_manifest.py` |
| `processed/scores_superseded/` | — | Score files kept out of the analysis on purpose. Currently the first DetectCodeGPT run, which used a log-likelihood statistic instead of the published normalised perturbed log rank |
| `processed/validity/` | `analysis/03_validity/` | RQ1 metrics, RQ2 strata, published operating points, calibration |
| `processed/correction/` | `analysis/04_correction/` | RQ3 Rogan–Gladen correction and estimability status |
| `processed/kg/` | `kg/build_graph.py` | The RDF graph the competency queries run against |
| `processed/llm_judge_cache/` | `analysis/02_detectors/llm_judge.py` | Cached judge responses keyed by model, prompt revision and text hash, so re-runs cost nothing |
| `processed/aidev_profile.json`, `repo_ages.parquet` | `profile_aidev.py`, `verify_repo_ages.py` | Dataset profile behind the corpus-description macros; per-repository creation dates behind the N1 frame |

## Preservation

The scoring runs are the one part of this project that cannot be regenerated cheaply:
about fourteen hours of H100 time, on a host that was decommissioned on 2026-08-14.
Before deletion, every file that host held was checksummed against the local copy and
matched byte for byte, and the two helper scripts that existed only there were retrieved
into `analysis/02_detectors/host/`.

`python analysis/make_release.py` bundles the artifacts that must outlive any single
machine into `release/scoring-artifacts-v1.tar.gz` with a SHA-256 manifest: the seven
score files, the superseded DetectCodeGPT scores, the RQ1 to RQ3 outputs, the RDF graph,
and the corpus and download manifests. Roughly 14 MB compressed. It is deposited
alongside the code so that every table in the paper can be reproduced without a GPU and
without the artifact text.

The bundle deliberately excludes `corpus_v1/corpus.parquet` and the `n1_*` files, which
carry repository text that we cannot redistribute, and the LLM-judge cache, which is
redundant with `llm_judge.parquet`. The exclusions and their reasons are recorded inside
the bundle's own manifest rather than only here.

## Versioning

The frozen corpus is tagged `corpus-v1` at the commit that froze it. Its checksum
manifest lives with the code in `analysis/01_corpus/manifests/`, not here, so that the
record survives even though the payloads are gitignored. A new freeze takes the next tag
and a new `processed/corpus_vN/` directory rather than overwriting this one, because
every score file is aligned to its corpus by row position.
