# Host-side helpers

These two scripts lived only on the H100 host and were retrieved before it was
decommissioned (2026-08-14). They are not imported by anything; they are kept because
they are the record of how the GPU sweep was actually driven, and because a replicator
setting up a new host would otherwise have to reinvent them.

- `dl_models.py` pre-fetches the scoring model weights so that a sweep can run with
  `HF_HUB_OFFLINE=1` and cannot silently pick up a different revision mid-run.
- `sweep.sh` drives the three GPU detectors in sequence.

Two corrections to `sweep.sh`, left in place rather than edited so that the historical
record stays intact: it lists `Qwen/Qwen2.5-Coder-7B` as a DetectCodeGPT model, which was
the first, incorrect implementation; the released scores use
`codellama/CodeLlama-7b-hf`, the base model of the reference implementation. It also
passes `--batch-size 16` for that detector, where the released scores were produced at
64. `HOST_SETUP.md` carries the table of what actually ran.
