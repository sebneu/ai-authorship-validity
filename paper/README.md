# Manuscript

Springer `svjour3` (EMSE format), one file per section.

```
main.tex                 preamble, front matter, \input list
sections/01..09          one file per section
generated/               numbers emitted by analysis/ -- never edited by hand
refs.bib                 only verified entries; see instructions/LITERATURE.md
check_manuscript.py      static checks (see below)
```

## Build

```bash
make
```

Regenerates the numbers, runs `latexmk`, then runs the checks and reports page count and
open placeholders. Other targets: `make check`, `make watch` (continuous preview),
`make clean`, and `make submit-check` (fails while any placeholder is open).

### Toolchain

TeX is a per-user **TinyTeX** install at `~/Library/TinyTeX` — no `sudo`, nothing written
to `/usr/local`. The Makefile puts it on `PATH` itself, so no shell setup is needed. To
reinstall on another machine:

```bash
curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh
```

The default package set compiled this document as-is: `svjour3` is vendored here, and
`booktabs`, `hyperref`, `natbib`, `amsmath` and `fix-cm` were all present. If a future
addition needs something else, `tlmgr install <pkg>` works without `sudo`.

Note that `natbib` is passed as a **class option** (`\documentclass[...,natbib]`), not
loaded with `\usepackage` — that is how `svjour3` binds `\citep`/`\citet` to Springer's
`spbasic` style, and loading it the usual way silently fails.

## Two rules the checker enforces

**Every corpus number comes from a macro.** `generated/corpus_numbers.tex` is written by
`analysis/01_corpus/profile_aidev.py --tex`. Change the data, re-run, and the prose
updates. Nothing measured is typed into a section file.

**Every borrowed number is marked.** Figures taken from other papers are wrapped in
`\citednum{...}`, which prints them unchanged but tells the checker they are not ours.
So each number in the manuscript is either a generated macro or explicitly someone
else's — there is no third category.

```bash
python check_manuscript.py            # macros resolve, inputs exist, no stray numbers
python check_manuscript.py --strict   # additionally: no open placeholders (pre-submission)
```

## Placeholders

`\RESULT{...}` marks anything still awaiting a number or a decision. It renders in bold
brackets so it cannot be missed in a PDF read-through.

```bash
grep -rn 'RESULT{' sections/ main.tex
```

Regenerate the numbers after any corpus change:

```bash
../.venv/bin/python ../analysis/01_corpus/profile_aidev.py --tex generated/corpus_numbers.tex
```
