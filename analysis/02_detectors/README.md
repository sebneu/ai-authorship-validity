# Detector harness

Scores every text in the frozen corpus with each detection approach under evaluation.
One adapter per detector, one row per (text, detector) pair.

## Contract

A detector takes text and returns a score. Nothing else.

```python
class Detector(Protocol):
    name: str
    version: str
    def score(self, texts: list[str]) -> list[float]: ...
```

Higher scores mean "more likely machine-generated". Adapters normalise direction so the
sign convention holds across detectors whose native statistics point different ways.

**Detectors see text and nothing else.** No agent label, repository, date, filename or
set membership reaches an adapter. `run_detectors.py` enforces this by passing a list of
strings rather than a dataframe, and scores are joined back on `source_id` afterwards.
The guard exists because most of the fields in the corpus correlate with the label: a
detector handed a 2019 timestamp does not need to read the text at all.

## Detectors

| Adapter | Kind | Hardware |
|---|---|---|
| `fast_detect_gpt` | zero-shot, conditional probability curvature | GPU |
| `binoculars` | zero-shot, perplexity / cross-perplexity ratio | GPU, two models resident |
| `detect_code_gpt` | zero-shot, code-specific perturbation | GPU |
| `heuristics` | phrase and formatting rules from the published studies | CPU |
| `llm_judge` | language model prompted to classify | GPU |
| `fingerprint` | supervised, 41 features, after Ghaleb (arXiv:2601.17406) | CPU |

`heuristics` and `fingerprint` run here. The rest need the H100/H200 host; see
`requirements-gpu.txt`.

Scoring models are Llama-3-8B class, following the Fast-DetectGPT authors' note that
Llama-3-8B outperforms falcon-7b as sampling and scoring model. No commercial detector
is included, so the whole evaluation reproduces from open weights.

## Why `fingerprint` is included

Ghaleb's classifier is the closest published work to this study. It discriminates among
five agents on contributions already known to be agent-authored, which is a different
task from ours, so it is not a competitor. Run here it estimates what a classifier
achieves when trained on the distribution it is tested on, giving an upper reference
point for the zero-shot detectors that the published estimates actually rely on.

## Running

```bash
python run_detectors.py --detectors heuristics            # CPU, runs anywhere
python run_detectors.py --detectors all --batch-size 32   # GPU host
python run_detectors.py --dry-run                         # list what would run
```

Output is `data/processed/scores/<detector>.parquet` with `source_id`, `detector`,
`version`, `score` and `elapsed_ms`. One file per detector so a failed run costs one
detector rather than the whole sweep, and so the GPU host and this machine can each
write their own without conflict.

## Reproducibility

Every adapter reports a `version` string covering its own revision and the model
weights it loaded. Seeds are fixed at the driver. Model identifiers are pinned in
`requirements-gpu.txt` rather than resolved at runtime, because a detector whose scoring
model silently changes produces numbers that cannot be reproduced, which is the failure
this paper is about.

## Open items

All three are closed. What replaced them:

1. **The four 2026 studies use no stylistic phrase list.** Two search for a
   contributor's own admission that a tool wrote the code, and that mechanism is
   reproduced in `selfadmission.py`; one applies the zero-shot stack; one classifies no
   text at all. `heuristics.py` keeps its formatting rules but is now named for what it
   is, a set of maintainer-style screening rules reproducing nobody's pipeline.
2. **DetectCodeGPT is pinned** to Shi et al., ICSE 2025 (arXiv:2401.06461), and
   reimplemented against the paper and the released code. The statistic is the
   normalised perturbed log rank; the earlier log-likelihood version scored an
   instrument nobody published and its results are archived unused.
3. **The LLM-judge prompt is frozen** at revision `p1`, fixed before any result was
   seen and quoted in the replication package. Prompt sensitivity is reported as a
   limitation rather than explored, since exploring it would measure our prompt
   engineering instead of the instrument.
