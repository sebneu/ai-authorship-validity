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
latexmk -pdf main.tex
```

There is no TeX installation on the development machine at time of writing, so **the
draft has not yet been compiled**. Before trusting the layout, install a toolchain:

```bash
brew install --cask basictex
```

BasicTeX needs `sudo tlmgr install` for anything beyond the minimal set; `svjour3` is
vendored here, but `booktabs`, `hyperref` and `fix-cm` may need pulling in. The
alternative is the full MacTeX cask (~4 GB, no extra package management).

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
