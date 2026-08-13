"""Formatting and phrasing markers of the kind maintainers screen on.

Reading the four 2026 measurement studies settled what this adapter is and is not. None
of them uses a stylistic rule list: two search for a developer's own admission that a
tool wrote the code (`selfadmission.py`), one applies the zero-shot detectors, and the
fourth estimates the contribution flood from pull request volume without classifying
any text. The rules below therefore reproduce no published pipeline.

They are kept because the markers are real and are applied informally. Maintainers,
triage bots and contribution policies screen on exactly these signals -- an emoji
header, a bold label list, an offer of further help -- and the paper's question is what
such a screen does to text that predates language models. Read the results as a
statement about these rules, named in the paper as such, and never as a measurement of
any study's method.

Scores are the count of matched rules divided by the number of rules, so the output is
in [0, 1] and comparable across texts of different length. Per-rule hits are available
through `explain()`, which matters because a rule that fires on almost everything is
worth reporting separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from _registry import register

VERSION = "heuristics/2026-08-03"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    note: str


def _rule(name: str, pattern: str, note: str, flags: int = re.IGNORECASE) -> Rule:
    return Rule(name, re.compile(pattern, flags), note)


# Assistant-voice phrasing. These are the surface markers most commonly cited, and the
# ones most likely to be edited out before a contribution is submitted, so they should
# have high precision and poor recall.
PHRASE_RULES = [
    _rule("assistant_voice", r"\b(as an ai|i'm an ai|language model)\b",
          "explicit self-identification"),
    _rule("offer_of_help", r"\b(let me know if|hope this helps|feel free to)\b",
          "conversational closing"),
    _rule("certainly", r"^(certainly|sure thing|absolutely)[!,]", "chat-style opener"),
    _rule("here_is", r"^here('s| is) (the|a|an|your)\b", "chat-style opener"),
    _rule("summary_offer", r"\b(in summary|to summarize|overall,)\b", "summarising register"),
]

# Formatting signatures. These fire on structure rather than wording and survive light
# editing, so they should behave the opposite way: better recall, worse precision.
FORMAT_RULES = [
    _rule("emoji", r"[\U0001F300-\U0001FAFF✨✅❌⚠]", "emoji present",
          flags=0),
    _rule("bold_headers", r"^\*\*[^*]+\*\*:?\s*$", "bolded section headings",
          flags=re.MULTILINE),
    _rule("markdown_heading", r"^#{1,4}\s+\S", "markdown headings", flags=re.MULTILINE),
    _rule("bullet_block", r"(?:^[-*]\s+.+\n){3,}", "three or more consecutive bullets",
          flags=re.MULTILINE),
    _rule("checkbox_list", r"^\s*[-*]\s+\[[ xX]\]", "task-list checkboxes",
          flags=re.MULTILINE),
    _rule("bold_label_line", r"^\s*[-*]\s+\*\*[^*]+\*\*\s*[:–-]",
          "bulleted bold label followed by prose", flags=re.MULTILINE),
]

# Structural conventions. Weak individually; included because studies cite them.
STRUCTURE_RULES = [
    _rule("conventional_commit", r"^(feat|fix|chore|docs|refactor|test|perf|build|ci)"
          r"(\([^)]+\))?!?:\s", "conventional-commit prefix"),
    _rule("triadic_list", r"\b\w+, \w+,? and \w+\b", "three-item list"),
]

RULES: list[Rule] = PHRASE_RULES + FORMAT_RULES + STRUCTURE_RULES


class HeuristicDetector:
    name = "heuristics"
    version = VERSION

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules or RULES

    def score(self, texts: list[str]) -> list[float]:
        n = len(self.rules)
        return [
            sum(1 for rule in self.rules if rule.pattern.search(t or "")) / n for t in texts
        ]

    def explain(self, texts: list[str]) -> list[dict[str, bool]]:
        """Per-rule hits, for reporting which rules carry the decision."""
        return [
            {rule.name: bool(rule.pattern.search(t or "")) for rule in self.rules}
            for t in texts
        ]


@register("heuristics", kind="heuristic", needs_gpu=False,
          note="formatting and phrasing markers; reproduces no published pipeline")
def _build() -> HeuristicDetector:
    return HeuristicDetector()
