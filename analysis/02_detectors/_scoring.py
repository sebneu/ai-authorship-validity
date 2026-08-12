"""Shared token-level scoring for the zero-shot detectors.

All three statistical detectors need quantities derived from a causal language model's
next-token distribution, which is why they cannot run against a chat API: that returns
sampled text or at best top-k logprobs, and these statistics are sums over the full
vocabulary.

Memory is the constraint. A batch of 8 texts at 1024 tokens with a 152k vocabulary is
2.5 GB of logits in bf16, and the derived quantities need several such tensors at once.
Everything below therefore reduces over the vocabulary in chunks along the sequence and
never holds more than one chunk of logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Texts are truncated to this many tokens. The corpus median is 86 tokens and the
# ninetieth percentile well under this, so truncation affects a small tail while
# bounding the worst case: one 178k-character diff would otherwise dominate a batch.
MAX_TOKENS = 1024

# Sequence positions per vocabulary reduction. Tuned so logits stay near 1 GB.
CHUNK = 128


@dataclass
class TokenStats:
    """Per-text sums over token positions, already reduced over the vocabulary."""

    logp_observed: torch.Tensor  # sum of log p(x_t | x_<t)
    mu: torch.Tensor  # sum of E_{v~p}[log p(v)]
    var: torch.Tensor  # sum of Var_{v~p}[log p(v)]
    n_tokens: torch.Tensor


class CausalScorer:
    """A loaded causal LM plus the reductions the detectors need."""

    def __init__(self, model_id: str, device: str = "cuda", dtype=torch.bfloat16) -> None:
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True
        ).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or "unpinned"

    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
        )
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.no_grad()
    def logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(**batch).logits.float()

    @torch.no_grad()
    def token_stats(self, texts: list[str]) -> TokenStats:
        """Observed log-probability plus the mean and variance of the log-probability
        under the model's own distribution, summed over positions.

        The last two are what makes Fast-DetectGPT analytic: they are the expectation
        and variance the original DetectGPT estimated by sampling perturbations.
        """
        batch = self.encode(texts)
        ids, mask = batch["input_ids"], batch["attention_mask"]
        logits = self.logits(batch)

        # Predict position t from t-1: drop the last logit and the first label.
        logits = logits[:, :-1, :]
        labels = ids[:, 1:]
        valid = mask[:, 1:].bool()

        b, s, _ = logits.shape
        logp_obs = torch.zeros(b, device=self.device)
        mu_sum = torch.zeros(b, device=self.device)
        var_sum = torch.zeros(b, device=self.device)

        for start in range(0, s, CHUNK):
            end = min(start + CHUNK, s)
            chunk = logits[:, start:end, :]
            lab = labels[:, start:end]
            ok = valid[:, start:end]

            logprobs = torch.log_softmax(chunk, dim=-1)
            probs = logprobs.exp()

            observed = logprobs.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
            mu = (probs * logprobs).sum(-1)
            second = (probs * logprobs.square()).sum(-1)
            var = (second - mu.square()).clamp_min(0)

            logp_obs += (observed * ok).sum(-1)
            mu_sum += (mu * ok).sum(-1)
            var_sum += (var * ok).sum(-1)
            del logprobs, probs, chunk

        return TokenStats(logp_obs, mu_sum, var_sum, valid.sum(-1).clamp_min(1))

    @torch.no_grad()
    def cross_entropy_against(
        self, other: "CausalScorer", texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Binoculars quantities: this model's own log-perplexity on the text, and the
        cross-entropy of this model's predicted distribution against `other`'s.

        Both models must tokenise identically, which is why Binoculars pairs a base
        model with its own instruct variant rather than two arbitrary models.
        """
        batch = self.encode(texts)
        ids, mask = batch["input_ids"], batch["attention_mask"]
        logits_self = self.logits(batch)[:, :-1, :]
        logits_other = other.logits(batch)[:, :-1, :]
        labels = ids[:, 1:]
        valid = mask[:, 1:].bool()

        b, s, _ = logits_self.shape
        nll = torch.zeros(b, device=self.device)
        xent = torch.zeros(b, device=self.device)

        for start in range(0, s, CHUNK):
            end = min(start + CHUNK, s)
            ok = valid[:, start:end]
            lp_self = torch.log_softmax(logits_self[:, start:end, :], dim=-1)
            lp_other = torch.log_softmax(logits_other[:, start:end, :], dim=-1)
            p_self = lp_self.exp()

            observed = lp_self.gather(-1, labels[:, start:end].unsqueeze(-1)).squeeze(-1)
            nll += (-observed * ok).sum(-1)
            xent += (-(p_self * lp_other).sum(-1) * ok).sum(-1)
            del lp_self, lp_other, p_self

        n = valid.sum(-1).clamp_min(1)
        return nll / n, xent / n, n

    def free(self) -> None:
        del self.model
        torch.cuda.empty_cache()
