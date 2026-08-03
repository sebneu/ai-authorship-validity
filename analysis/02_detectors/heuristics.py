"""Heuristic phrase and formatting filters.

Several published estimates of AI-authored contributions rest on rules of this kind
rather than on a statistical detector: a phrase list, a formatting signature, a
structural pattern. This adapter reimplements the mechanism so those rules can be
evaluated on the same footing as the model-based detectors.

The rule set below is a documented default, not a reproduction of any particular study.
Extracting the exact lists from the four 2026 studies is an open item (see this
directory's README). Until that is done, results from this adapter describe these rules
and not those papers, and the distinction has to survive into the write-up.

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
          note="phrase and formatting rules; default set pending extraction from the studies")
def _build() -> HeuristicDetector:
    return HeuristicDetector()
