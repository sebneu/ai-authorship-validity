# Who Wrote This? A Validity Audit of AI-Authorship Attribution in Open-Source Software Artifacts

Replication package for the paper of the same name, submitted to *Empirical Software
Engineering*, Special Issue "Agentic Software Engineering: The Rise of AI Teammates".

**Sebastian Neumaier** ([0000-0002-9804-4882](https://orcid.org/0000-0002-9804-4882)) ·
University of Applied Sciences St. Pölten · Vienna University of Economics and Business

Archived release: [10.5281/zenodo.21976613](https://doi.org/10.5281/zenodo.21976613)

## What the paper does

A growing literature estimates how much of open-source software is written by AI, using
zero-shot detectors, keyword scans and language models as classifiers. None of those
instruments had been validated against ground truth on the artifacts they are applied
to. This paper measures what they do there.

Ground truth needs no human annotation. Contributions opened by coding agents under
their own accounts give a positive set; artifacts written in the same repositories
before ChatGPT was released give a negative set in which every detection is an error by
construction. The measured error rates are then propagated through the Rogan–Gladen
correction from epidemiology.

The headline results: at the operating point published for software artifacts,
Binoculars marks 98.9% of pre-ChatGPT code diffs as AI-written. Calibrated to a five per
cent error budget, the zero-shot detectors rank pre-ChatGPT human text as more
machine-written than agent text in every artifact type tested. A classifier trained
in-domain on the same input reaches 0.982 AUROC on pull request descriptions, so the
signal is present and the deployed instruments miss it. For most instrument and
artifact-type pairs, no true prevalence between zero and one could have produced the
rates the literature reports, so those estimates need remeasuring rather than adjusting.

## Layout

```
analysis/
  01_corpus/      ground-truth corpus construction and its manifests
  02_detectors/   detector harness: one adapter per instrument, version-pinned
  03_validity/    RQ1 metrics, RQ2 strata, published operating points, calibration
  04_correction/  RQ3 Rogan-Gladen correction
  run_all.py      regenerates every table and figure
kg/               aiprov provenance vocabulary, RDF graph, SPARQL competency queries
paper/            LaTeX source, generated tables and figures, consistency check
data/README.md    provenance record for every input and output
```

## Reproducing the results

```bash
pip install -r requirements.txt
python analysis/run_all.py     # everything downstream of the detector score files
make -C paper                  # manuscript, with the consistency check
```

Corpus construction and GPU scoring are not in the runner: they need credentials or an
H100, and neither changes except by a deliberate decision. `analysis/02_detectors/HOST_SETUP.md`
records the hardware, the pinned model revisions and the batch sizes that produced the
released scores.

Every number in the manuscript is written into the LaTeX source by the analysis code as
a macro, and `paper/check_manuscript.py` fails the build if the text contains a
hand-typed figure where a generated one belongs.

## What is not included, and why

Repository text is not redistributed: the licences of several thousand projects do not
allow it, and a static copy would hide how each cell was selected. The corpus is
released as the specification instead — artifact identifiers, selection predicates,
seeds, per-cell counts and the code that rebuilds it. The AIDev dataset is not mirrored
either; the exact revision is pinned so that counts reproduce.

The Zenodo release adds what cannot be regenerated without a GPU: the seven score files,
the corpus metadata without its text column, and the RQ1–RQ3 outputs. With those, every
table in the paper can be recomputed on a laptop. `data/README.md` records what each
file is and where it came from.

## Citation

```bibtex
@article{neumaier2026whowrotethis,
  title   = {Who Wrote This? A Validity Audit of {AI}-Authorship Attribution in
             Open-Source Software Artifacts},
  author  = {Neumaier, Sebastian},
  journal = {Empirical Software Engineering},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code under `analysis/` and `kg/` is MIT. Manuscript text, figures and the `aiprov`
vocabulary are CC BY 4.0. See `LICENSE`.

## Funding

Supported by the Internet Stiftung through its netidee programme, project
[CrOSSD2](https://www.netidee.at/crossd2), and by the European Union's Horizon Europe
programme under Grant Agreement No. 101189650
([CERTAIN](https://certain-project.eu/)). Views and opinions expressed are those of the
author only and do not necessarily reflect those of the European Union. Neither the
European Union nor the granting authority can be held responsible for them.
