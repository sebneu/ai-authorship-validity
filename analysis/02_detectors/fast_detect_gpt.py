"""Fast-DetectGPT: conditional probability curvature.

Bao et al. (ICLR 2024). DetectGPT observed that machine-generated text sits at a local
maximum of the generating model's likelihood surface, and estimated the curvature by
perturbing the text and rescoring, which costs a hundred forward passes. Fast-DetectGPT
replaces the sampling with the analytic conditional mean and variance of the token
log-probability under the model's own distribution, so one pass suffices.

The statistic is

    d(x) = ( log p(x) - E_{x~p}[log p(x)] ) / sqrt( Var_{x~p}[log p(x)] )

summed over token positions. Text the model itself would have produced scores high;
human text scores near zero or below.

We use one model for both sampling and scoring, which the authors show is the stronger
setting and which halves the memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _env import get  # noqa: E402
from _registry import register  # noqa: E402
from _scoring import CausalScorer  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-7B"


class FastDetectGPT:
    name = "fast_detect_gpt"

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or get("FASTDETECT_MODEL") or DEFAULT_MODEL
        self.scorer = CausalScorer(self.model_id)
        self.version = f"fast_detect_gpt/{self.model_id}@{self.scorer.revision[:12]}"

    def score(self, texts: list[str]) -> list[float]:
        stats = self.scorer.token_stats(texts)
        # Normalised curvature. The clamp guards texts of one or two tokens, where the
        # variance is degenerate and the ratio would otherwise blow up.
        denominator = stats.var.sqrt().clamp_min(1e-6)
        d = (stats.logp_observed - stats.mu) / denominator
        return d.float().cpu().tolist()


@register("fast_detect_gpt", kind="zero-shot", needs_gpu=True,
          note="analytic curvature, one forward pass per text")
def _build() -> FastDetectGPT:
    return FastDetectGPT()
