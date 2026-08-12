"""Binoculars: perplexity against cross-perplexity.

Hans et al. (ICML 2024). Perplexity alone confuses "predictable text" with "machine
text": a formulaic bug report is low-perplexity because it is formulaic. Binoculars
divides the text's perplexity under one model by the cross-perplexity between two
closely related models, which normalises away how intrinsically predictable the text is
and leaves how well it matches what a model would produce.

    B(x) = logPPL_observer(x) / X-PPL(observer, performer, x)

Low B means machine-generated, so the returned score is negated to keep the harness
convention that higher means more likely machine-generated.

The two models must share a tokenizer, which is why the pair is a base model and its
own instruct variant rather than two unrelated models.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _env import get  # noqa: E402
from _registry import register  # noqa: E402
from _scoring import CausalScorer  # noqa: E402

# The pair from the original paper. Both are ungated, which keeps the replication
# package installable without accepting a licence.
DEFAULT_OBSERVER = "tiiuae/falcon-7b"
DEFAULT_PERFORMER = "tiiuae/falcon-7b-instruct"


class Binoculars:
    name = "binoculars"

    def __init__(self, observer: str | None = None, performer: str | None = None) -> None:
        self.observer_id = observer or get("BINOCULARS_OBSERVER") or DEFAULT_OBSERVER
        self.performer_id = performer or get("BINOCULARS_PERFORMER") or DEFAULT_PERFORMER
        self.observer = CausalScorer(self.observer_id)
        self.performer = CausalScorer(self.performer_id)

        if self.observer.tokenizer.get_vocab() != self.performer.tokenizer.get_vocab():
            raise RuntimeError(
                f"{self.observer_id} and {self.performer_id} do not share a tokenizer. "
                "Cross-perplexity compares distributions position by position, so the "
                "two models must segment the text identically."
            )
        self.version = (
            f"binoculars/{self.observer_id}+{self.performer_id}"
            f"@{self.observer.revision[:8]}"
        )

    def score(self, texts: list[str]) -> list[float]:
        ppl, xppl, _ = self.observer.cross_entropy_against(self.performer, texts)
        b = ppl / xppl.clamp_min(1e-6)
        # Negated: the paper's statistic is low for machine text, the harness expects
        # high. Doing it here keeps every detector on one convention.
        return (-b).float().cpu().tolist()


@register("binoculars", kind="zero-shot", needs_gpu=True,
          note="two models resident, ~40 GB VRAM")
def _build() -> Binoculars:
    return Binoculars()
