"""DetectCodeGPT: normalised perturbed log rank under stylised code perturbation.

Where DetectGPT perturbs prose by masking spans and refilling them with T5, code offers
a cheaper handle: machine-written code is unusually regular in its whitespace, so
inserting spaces and newlines pushes it further down the model's preference ordering
than it pushes human-written code. The statistic is the normalised perturbed log rank

    NPR(x) = ( 1/k * sum_i logrank(x~_i) ) / logrank(x)

where logrank is the mean over tokens of the log of the observed token's rank in the
model's likelihood ordering. Higher means more likely machine-written.

Parameters follow Shi et al. (ICSE 2025, arXiv:2401.06461) and their released
implementation at github.com/YerbaPage/DetectCodeGPT. Three points where the two
disagree, or where the paper is silent, are resolved here and recorded in the version
string so that a later correction shows up in the results rather than silently:

  * The paper's Algorithm 1 subtracts a mean of NPR terms, which contradicts its own
    Equation 2 and the released code. Equation 2 is implemented.
  * The paper gives lambda_newlines = 2, but the released code inserts exactly one
    newline per selected line; its Poisson draw is commented out. The code is followed,
    since it is the artifact that produced the published numbers.
  * Spaces are drawn as Poisson(span_length) + 1 with span_length = 2, whose mean of 3
    matches the paper's stated lambda_spaces = 3. Paper and code agree here.

This is by far the most expensive detector in the harness: k+1 forward passes per text
against one for the others. It runs on diffs alone, which is what the method was built
for.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from _env import get  # noqa: E402
from _registry import register  # noqa: E402
from _scoring import CausalScorer  # noqa: E402

# The base model of the reference implementation. Ji et al. (arXiv:2607.01867) apply
# DetectCodeGPT to repository code with these defaults, and it is their deployment of
# the instrument that this study audits, so the audit uses the instrument as published
# rather than a stronger surrogate.
DEFAULT_MODEL = "codellama/CodeLlama-7b-hf"

# k in the paper; 50 in every reported experiment. Their Table VI shows the score
# flattening from 20 perturbations upward, so this is well inside the stable region.
DEFAULT_PERTURBATIONS = 50

# Share of insertion sites that receive a perturbation (alpha and beta in the paper,
# pct_words_masked in the released configuration).
PCT = 0.5

# Poisson mean for the space run length; the drawn value is incremented by one.
SPAN_LENGTH = 2


def _seed(text: str, index: int) -> np.random.Generator:
    """A generator fixed by the text and the perturbation index.

    Seeding per text rather than per run keeps a score independent of the batch it was
    scored in, so re-running one genre or resuming an interrupted sweep reproduces the
    earlier numbers exactly.
    """
    digest = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest()
    return np.random.default_rng((int.from_bytes(digest, "big") << 8) | (index & 0xFF))


def insert_spaces(text: str, rng: np.random.Generator) -> str:
    tokens = text.split(" ")
    n = len(tokens)
    chosen = rng.choice(n, int(n * PCT), replace=False)
    for idx in chosen:
        tokens[idx] = tokens[idx] + " " * (rng.poisson(SPAN_LENGTH) + 1)
    return " ".join(tokens)


def insert_newlines(text: str, rng: np.random.Generator) -> str:
    lines = text.split("\n")
    n = len(lines)
    chosen = rng.choice(n, int(n * PCT), replace=False)
    for idx in chosen:
        lines[idx] = lines[idx] + "\n"
    return "\n".join(lines)


class DetectCodeGPT:
    name = "detect_code_gpt"

    def __init__(self, model_id: str | None = None, n_perturb: int | None = None) -> None:
        self.model_id = model_id or get("DETECTCODEGPT_MODEL") or DEFAULT_MODEL
        env_n = get("DETECTCODEGPT_N")
        self.n_perturb = n_perturb or (int(env_n) if env_n else DEFAULT_PERTURBATIONS)
        self.scorer = CausalScorer(self.model_id)
        self.version = (
            f"detect_code_gpt/npr/{self.model_id}@{self.scorer.revision[:8]}"
            f"/k{self.n_perturb}/pct{PCT}/span{SPAN_LENGTH}/nl1"
        )

    def perturb(self, text: str, index: int) -> str:
        """One perturbed copy. The first half of the copies receive spaces and the
        second half newlines, which is what the released code's even split of a
        replicated batch amounts to per text, and what the paper describes as choosing
        a perturbation type at random with probability one half.
        """
        rng = _seed(text, index)
        if index < self.n_perturb // 2:
            return insert_spaces(text, rng)
        return insert_newlines(text, rng)

    def score(self, texts: list[str]) -> list[float]:
        clean = [t or "" for t in texts]
        base = self.scorer.log_rank(clean).float()

        total = torch.zeros(len(clean), device=base.device)
        for i in range(self.n_perturb):
            perturbed = [self.perturb(t, i) for t in clean]
            total += self.scorer.log_rank(perturbed).float()
        mean_perturbed = total / self.n_perturb

        # A text whose tokens are all rank one has zero log rank and no defined NPR.
        # It cannot arise for a real diff, but a blank row would otherwise divide by
        # zero and poison the column silently.
        npr = torch.where(
            base > 0, mean_perturbed / base.clamp_min(1e-6), torch.full_like(base, float("nan"))
        )
        return npr.cpu().tolist()


@register("detect_code_gpt", kind="zero-shot", needs_gpu=True,
          note="k+1 passes per text; diffs only, roughly 10 h for the diff genre")
def _build() -> DetectCodeGPT:
    return DetectCodeGPT()
