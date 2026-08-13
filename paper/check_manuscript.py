#!/usr/bin/env python3
"""Static checks on the manuscript source.

Not a substitute for compiling, but catches the failures that cost the most time:
a macro used before it is generated, a missing \\input, a number typed into the prose
instead of pulled from analysis/, and -- near submission -- placeholders left open.

Usage:
    python check_manuscript.py
    python check_manuscript.py --strict     # also fail on open RESULT placeholders
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
GENERATED_DIR = PAPER / "generated"
# Every generator writes here, and each may define macros. Reading the directory rather
# than one file means a new generator does not have to be registered in two places, and
# a macro defined by any of them counts as defined.
GENERATED = GENERATED_DIR / "corpus_numbers.tex"

# Defined by the manuscript preamble rather than by the generator.
PREAMBLE_MACROS = {"RESULT"}

# Control sequences that belong to LaTeX or loaded packages, not to us.
LATEX_MACROS = {"RequirePackage"}

# Digit groups in prose that are almost certainly hand-typed corpus counts. Years,
# section numbers and small integers are fine; four-plus digits or comma groups are not.
SUSPICIOUS_NUMBER = re.compile(r"(?<![\\\w])(\d{1,3}(?:[,{]\d{3})+|\d{4,})(?![}\w])")

# A bare four-digit number in this range reads as a year, not as a measurement.
YEAR_RANGE = range(1900, 2101)

# Design parameters we chose, as opposed to quantities we measured. These are stated
# in the prose deliberately and have no place in the generated macros.
ALLOWED_NUMBERS = {"1000"}

# Filler and assistant-prose tells. The rule is in CLAUDE.md; enforcing it here means a
# lapse is caught by the build rather than by a reader. "prevalence" is deliberately
# absent: it is the technical term the paper is about.
BANNED = [
    "delve", "underscore", "leverage", "facilitate", "notably", "importantly",
    "it is worth", "should be noted", "furthermore", "moreover", "in conclusion",
    "robust", "comprehensive", "seamless", "paradigm", "cutting-edge", "transformative",
    "holistic", "significantly", "various", "numerous", "a wide range of",
    "plays a crucial role", "serves as", "empowers", "demonstrates the importance",
    "highlights", "sheds light on", "utilize", "prior to", "subsequent to",
    "due to the fact",
]


def sources() -> list[Path]:
    return sorted(PAPER.glob("sections/*.tex")) + [PAPER / "main.tex"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    problems: list[str] = []
    warnings: list[str] = []

    if not GENERATED.exists():
        print(
            f"missing {GENERATED.relative_to(PAPER)}\n"
            "run: python ../analysis/01_corpus/profile_aidev.py "
            "--tex generated/corpus_numbers.tex",
            file=sys.stderr,
        )
        return 2

    defined: set[str] = set()
    for path in sorted(GENERATED_DIR.glob("*.tex")):
        defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", path.read_text()))
    known = defined | PREAMBLE_MACROS | LATEX_MACROS

    main_text = (PAPER / "main.tex").read_text()
    for target in re.findall(r"\\input\{([^}]+)\}", main_text):
        path = PAPER / (target if target.endswith(".tex") else target + ".tex")
        if not path.exists():
            problems.append(f"missing \\input target: {target}")

    for path in sources():
        text = path.read_text()
        rel = path.relative_to(PAPER)

        for macro in set(re.findall(r"\\([A-Z][A-Za-z]+)\b", text)):
            if macro not in known:
                problems.append(f"{rel}: undefined macro \\{macro}")

        for line_no, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            # Numbers inside a RESULT placeholder are notes to self, not claims, and
            # \citednum marks a figure taken from another paper.
            stripped = re.sub(r"\\RESULT\{[^}]*\}", "", line)
            stripped = re.sub(r"\\citednum\{[^}]*\}", "", stripped)
            for number in SUSPICIOUS_NUMBER.findall(stripped):
                bare = number.replace(",", "").replace("{", "")
                if bare in ALLOWED_NUMBERS:
                    continue
                if bare.isdigit() and int(bare) in YEAR_RANGE and len(bare) == 4:
                    continue
                warnings.append(
                    f"{rel}:{line_no}: hand-typed number '{number}' -- "
                    "add a macro in profile_aidev.py instead"
                )

            lowered = stripped.lower()
            for phrase in BANNED:
                if re.search(r"(?<![a-z])" + re.escape(phrase), lowered):
                    warnings.append(f"{rel}:{line_no}: banned expression '{phrase}'")

    # Count uses, not the \newcommand that defines it.
    open_results = sum(len(re.findall(r"\\RESULT\{", p.read_text())) for p in sources())

    for w in warnings:
        print(f"warning: {w}")
    for p in problems:
        print(f"ERROR: {p}")

    print(f"\n{len(defined)} generated macros, {open_results} open RESULT placeholders")

    if args.strict and open_results:
        print("ERROR: placeholders remain (--strict)", file=sys.stderr)
        return 1
    if problems:
        return 1
    print("ok" if not warnings else "ok, with warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
