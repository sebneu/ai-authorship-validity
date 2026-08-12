"""DetectCodeGPT: perturbation discrepancy with code-specific perturbations.

Where DetectGPT perturbs prose by masking and refilling spans with T5, code has a
cheaper handle: model-written code is unusually regular in its whitespace, so inserting
spaces and newlines moves machine-written code further down the likelihood surface than
human-written code. The statistic is the normalised discrepancy

    d(x) = ( log p(x) - mean_i log p(x~_i) ) / std_i log p(x~_i)

over N perturbed copies.

This is by far the most expensive detector here: N+1 forward passes per text against one
for the others. N is therefore configurable, and the cost of changing it is stated
rather than hidden -- at the default it accounts for most of a full sweep.

The exact perturbation scheme and N of the original paper still need confirming against
the publication (see instructions/LITERATURE.md, section B). The implementation below
follows the described method; the parameters are recorded in the version string so a
later correction is visible in the results rather than silent.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _env import get  # noqa: E402
from _registry import register  # noqa: E402
from _scoring import CausalScorer  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B"
DEFAULT_PERTURBATIONS = 50

# Fraction of eligible positions perturbed per copy.
SPACE_RATE = 0.05
NEWLINE_RATE = 0.02

LINE_START = re.compile(r"^", re.MULTILINE)


class DetectCodeGPT:
    name = "detect_code_gpt"

    def __init__(self, model_id: str | None = None, n_perturb: int | None = None) -> None:
        self.model_id = model_id or get("DETECTCODEGPT_MODEL") or DEFAULT_MODEL
        env_n = get("DETECTCODEGPT_N")
        self.n_perturb = n_perturb or (int(env_n) if env_n else DEFAULT_PERTURBATIONS)
        self.scorer = CausalScorer(self.model_id)
        self.version = (
            f"detect_code_gpt/{self.model_id}@{self.scorer.revision[:8]}"
            f"/n{self.n_perturb}"
        )
        self.rng = random.Random(7)

    def _perturb(self, text: str) -> str:
        """Insert spaces after tokens and blank lines between lines."""
        chars = list(text)
        out = []
        for ch in chars:
            out.append(ch)
            if ch in " \t" and self.rng.random() < SPACE_RATE:
                out.append(" ")
        perturbed = "".join(out)
        lines = perturbed.split("\n")
        with_blanks = []
        for line in lines:
            with_blanks.append(line)
            if self.rng.random() < NEWLINE_RATE:
                with_blanks.append("")
        return "\n".join(with_blanks)

    def score(self, texts: list[str]) -> list[float]:
        original = self.scorer.token_stats(texts)
        # Per-token so that a perturbation changing the length does not shift the score
        # through length alone.
        base = (original.logp_observed / original.n_tokens).float()

        samples = torch.zeros(self.n_perturb, len(texts), device=base.device)
        for i in range(self.n_perturb):
            perturbed = [self._perturb(t or "") for t in texts]
            stats = self.scorer.token_stats(perturbed)
            samples[i] = (stats.logp_observed / stats.n_tokens).float()

        mean = samples.mean(0)
        std = samples.std(0).clamp_min(1e-6)
        return ((base - mean) / std).cpu().tolist()


@register("detect_code_gpt", kind="zero-shot", needs_gpu=True,
          note="N+1 passes per text; dominates a full sweep, set DETECTCODEGPT_N to trade")
def _build() -> DetectCodeGPT:
    return DetectCodeGPT()
