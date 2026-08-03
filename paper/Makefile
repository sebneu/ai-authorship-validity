# Manuscript build.
#
# TeX lives in a per-user TinyTeX install (no sudo, nothing in /usr/local), so the
# path is set here rather than assumed to be on yours.

TEXBIN  := $(HOME)/Library/TinyTeX/bin/universal-darwin
LATEXMK := PATH="$(TEXBIN):$$PATH" latexmk

PYTHON  := ../.venv/bin/python
NUMBERS := generated/corpus_numbers.tex

.PHONY: all check clean distclean numbers watch

all: numbers
	$(LATEXMK) -pdf -interaction=nonstopmode main.tex
	@echo
	@$(PYTHON) check_manuscript.py
	@echo "pages: $$(pdfinfo main.pdf | awk '/^Pages/{print $$2}')"
	@echo "open placeholders: $$(grep -ro 'RESULT{' sections main.tex | wc -l | tr -d ' ')"

# Corpus counts come from the data, never from the keyboard.
numbers: $(NUMBERS)

$(NUMBERS): ../analysis/01_corpus/profile_aidev.py
	$(PYTHON) ../analysis/01_corpus/profile_aidev.py --tex $(NUMBERS)

check:
	@$(PYTHON) check_manuscript.py

# Fails if any placeholder is still open. Run before submitting.
submit-check: all
	$(PYTHON) check_manuscript.py --strict

watch: numbers
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode main.tex

clean:
	$(LATEXMK) -c
	rm -f main.bbl

distclean: clean
	rm -f main.pdf
