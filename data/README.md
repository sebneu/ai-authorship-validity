# Data

Payloads in this directory are **gitignored**. This file is the provenance record — every
dataset used by `analysis/` must be listed here with enough detail to reconstruct it.

| Path | Source | Version / accessed | Notes |
|---|---|---|---|
| `raw/aidev/` | AIDev dataset (arXiv:2507.15003), HF `hao-li/AIDev`, Zenodo 10.5281/zenodo.16919272 | _tbd_ | Positive ground truth (agent-authored PRs) |
| `raw/github/` | GitHub REST/GraphQL API, GH Archive | _tbd_ | Negative set (pre-2022-11-30, same repos), `Co-authored-by` trailers |
| `raw/crossd/` | CrOSSD panel (health.crossd.tech) | _tbd_ | Stratification; prevalence re-estimation sample |
| `processed/` | produced by `analysis/01_corpus/` | — | Frozen corpus versions, tagged |

Frozen corpus versions get a git tag (`corpus-v1`, ...) and a checksum manifest committed
under `analysis/01_corpus/manifests/`.
