# Manuscript build.
#
# TeX lives in a per-user TinyTeX install (no sudo, nothing in /usr/local), so the
# path is set here rather than assumed to be on yours.

TEXBIN  := $(HOME)/Library/TinyTeX/bin/universal-darwin
LATEXMK := PATH="$(TEXBIN):$$PATH" latexmk

PYTHON  := ../.venv/bin/python
NUMBERS := generated/corpus_numbers.tex

.PHONY: all check clean distclean numbers watch titlepage submission

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

# Editorial Manager wants author identity in a file of its own.
titlepage:
	$(LATEXMK) -pdf -interaction=nonstopmode title_page.tex
	@echo "unfilled fields: $$(grep -c 'TOADD{' title_page.tex)"

# Everything the submission needs, built together.
submission: all titlepage
	@echo
	@echo "upload to Editorial Manager:"
	@echo "  manuscript  paper/main.pdf"
	@echo "  title page  paper/title_page.pdf"
