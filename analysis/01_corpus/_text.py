"""Text normalisation shared by the positive and negative builders.

Both sides of the corpus must be normalised identically. If trailers were stripped from
pre-LLM commit messages but left in agent-authored ones, a detector could separate the
classes by reading a label rather than by judging the text -- and the measured accuracy
would be an artefact of the pipeline.
"""

from __future__ import annotations

import re

# Written with "-" as the word separator and expanded below to accept whitespace too:
# git's hyphenated convention and Phabricator's "Reviewed By:" are the same trailer.
# The Co- prefix is optional -- bare "Authored-by:" is the pair-programming convention
# at several shops and is just as much a declaration.
_TRAILER_WORDS = (
    "co?-authored-by|signed-off-by|reviewed-by|acked-by|tested-by|helped-by|"
    "reported-by|suggested-by|generated-by|assisted-by|noticed-by|co-developed-by|"
    "co-committed-by|on-behalf-of|pull-request-author"
)
TRAILER_NAMES = _TRAILER_WORDS.replace("-", "[-\\s]").replace("co?[-\\s]", "(?:co[-\\s]?)?")

# Trailers do not reliably sit at the start of a line: they appear indented inside
# squashed-merge bodies and appended mid-sentence by tools that omit the newline.
TRAILER = re.compile(rf"^\s*(?:{TRAILER_NAMES}):", re.IGNORECASE)
TRAILER_INLINE = re.compile(rf"\s*(?:{TRAILER_NAMES}):.*$", re.IGNORECASE)

# Automation accounts, identified from the author string. Matches become negative set
# N3 (pre-LLM machine-generated text) rather than being discarded: measuring how often
# detectors flag decade-old automation as AI is the point of N3.
BOT_PATTERNS = re.compile(
    r"(\[bot\]|dependabot|renovate|greenkeeper|snyk-bot|imgbot|allcontributors|"
    r"github-actions|semantic-release|release-please|mergify|pyup|whitesource|"
    r"scala-steward|depfu|codecov|travis|appveyor|circleci)",
    re.IGNORECASE,
)

# Workflow boilerplate with no analogue in the 2025 positives. Left in place a detector
# can learn "Phabricator block implies pre-2022 implies human", which is an era marker
# rather than an authorship signal. Detected here so its prevalence can be measured;
# whether to strip it is a corpus decision, not a parsing one.
BOILERPLATE = re.compile(
    r"^\s*(Summary|Test Plan|Differential Revision|Change-Id|Reviewers|Subscribers|"
    r"Reviewed on|Bug|Fixes|Refs|Pull-request|Auto-Submit|Commit-Queue|Cr-Commit-Position|"
    r"git-svn-id|Former-commit-id|Sponsored by|Upstream-Status|BUG|ISSUE):",
    re.IGNORECASE | re.MULTILINE,
)


def strip_trailers(message: str) -> tuple[str, bool]:
    """Remove trailer declarations. Returns (text a detector sees, had_trailer)."""
    kept, found = [], False
    for line in message.splitlines():
        if TRAILER.match(line):
            found = True
            continue  # the whole line is a declaration
        cleaned = TRAILER_INLINE.sub("", line)
        if cleaned != line:
            found = True
        if cleaned.strip():
            kept.append(cleaned)
    return "\n".join(kept).strip(), found


def has_boilerplate(message: str) -> bool:
    return bool(BOILERPLATE.search(message))


def is_bot_author(author: str | None) -> bool:
    return bool(author and BOT_PATTERNS.search(author))
