# Who Wrote This? A Validity Audit of AI-Authorship Attribution in Open-Source Software Artifacts

Research repository for a single-author empirical study submitted to *Empirical Software
Engineering* (EMSE), Special Issue "Agentic Software Engineering: The Rise of AI Teammates".

**Author:** Sebastian Neumaier

## What this is

Recent studies estimate how much of open-source software is AI-generated, using detectors
(Fast-DetectGPT, Binoculars, DetectCodeGPT), heuristic phrase filters, and LLM classifiers —
none of which have been validated against ground truth *in the OSS-artifact domain*. This
project inverts the question and audits the measurement instruments themselves, using
agent-attributed contributions as a high-precision positive set and pre-ChatGPT artifacts as
a near-certain negative set, then re-estimates published prevalence figures under the
measured error rates.

## Layout

```
analysis/        one script per table/figure; every empirical claim is regenerable
  01_corpus/     ground-truth corpus construction (positives, negatives, bot lists)
  02_detectors/  unified detector harness (text in -> score out), version-pinned
  03_validity/   RQ1 metrics, RQ2 stratified error analysis, counterfactual FPR
  04_correction/ RQ3 Rogan-Gladen prevalence correction + bootstrap
kg/              RQ4 provenance model (PROV-O extension), RDF dump, SPARQL queries
paper/           LaTeX manuscript (Springer svjour3), bibliography, figures
data/            raw + processed data (gitignored; provenance in data/README.md)
```

## Conventions

- Analysis in Python, reproducible, fixed seeds, pinned versions.
- No number enters the paper that is not produced by a script in `analysis/`.
- Target: public replication package (EMSE open-science badges).

## License

Code: MIT. Paper text and figures: CC BY 4.0 (see `LICENSE`).
