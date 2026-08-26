# GPU host requirements

> **Decommissioned 2026-08-14.** The H100 host this describes no longer exists, and its
> data was retrieved and verified before deletion: every file it held was checksummed
> against the local copy, and the score files match byte for byte. Nothing below is
> needed to reproduce the *results* -- the score files are archived and released -- but
> it is kept because it is what a replicator would need in order to re-score the corpus
> from scratch, and because the model revisions and batch sizes recorded here are part
> of what makes the numbers reproducible.
>
> The two helper scripts that lived only on that host are now in `host/`.

What the detector runs need from the VM, with the reasoning, so the sysadmin can size
it without a back-and-forth.

## GPU

**Allocated: one H100 as a vGPU, profile `H100L-94C`, driver 595.71.05, CUDA 13.2.**
Two weeks of access from 2026-08-04.

That covers the requirement with room to spare. 94 GB of the card is assigned to the
guest, `MIG M.` reports `N/A` so there is no partitioning, and the whole 94 GB is
available to one process. Binoculars needs about 40 GB for two resident models, so
nothing here is close to the limit. NVLink, FP64 and multi-GPU are not used.

**The earlier Blackwell caveat does not apply.** An H100 is Hopper, compute capability
9.0 (`sm_90`), which every PyTorch build since 2.0 supports. The `sm_120` problem was
specific to RTX PRO 6000.

Driver 595 advertises CUDA 13.2, while the pinned PyTorch is built against CUDA 12.8.
That combination is fine: NVIDIA drivers stay backward compatible with older CUDA
runtimes, so a cu128 build runs on a CUDA 13 driver. Verify it once rather than assume
it (see below) — the failure mode, if any, appears at the first `.cuda()` call, not at
install time.

Requirements, restated for the record:

- NVIDIA driver 525 or newer (595.71.05 far exceeds this)
- PyTorch built for cu126 or cu128; do not build against CUDA 13 unless a wheel is
  actually available for the pinned version

### Verify before the first long run

```bash
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('device', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
print('free/total GiB', [round(x/2**30, 1) for x in torch.cuda.mem_get_info()])
x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
print('matmul ok', (x @ x).sum().item() == (x @ x).sum().item())
"
```

Expect capability `(9, 0)` and roughly 94 GiB total. If the matmul raises *"no kernel
image is available"*, the wheel and the driver disagree and the CUDA build needs
changing; nothing else in the setup is worth doing until this passes.

## VM sizing

The proposed 4 GB RAM / 4 cores / 16 GB disk will not work. Concrete requirements:

| Resource | Proposed | Needed | Why |
|---|---|---|---|
| RAM | 4 GB | **64 GB** (min 32) | A 7B model in bf16 is ~15 GB of weights, staged through host memory on load. Binoculars holds two at once. 4 GB cannot load one. |
| vCPU | 4 | **8–16** | Tokenising 31 M tokens is the CPU-side bottleneck. 4 cores work but leave the GPU waiting. |
| Disk | 16 GB | **250 GB** (min 120) | Model weights alone are ~58 GB; see below. |

### Where the disk goes

| Item | Size |
|---|---|
| `Qwen/Qwen2.5-7B` | ~15 GB |
| `tiiuae/falcon-7b` | ~14 GB |
| `tiiuae/falcon-7b-instruct` | ~14 GB |
| `Qwen/Qwen2.5-Coder-7B` | ~15 GB |
| PyTorch + CUDA runtime | ~12 GB |
| OS, Python environment | ~10 GB |
| Corpus and outputs | ~1 GB |
| Headroom | ~30 GB |

**The data is not the problem.** Only `corpus.parquet` needs to reach the host, and it
is 44 MB. Everything else stays on the workstation.

## Access

- Shell access with the GPU visible (`nvidia-smi` works from the login shell).
- Outbound HTTPS to `huggingface.co`, or the four models pre-downloaded into a shared
  `HF_HOME`.
- No inbound ports needed. The job writes parquet files and exits.

## Runtime

Measured or derived against the frozen corpus (86,484 texts, 31.3 M tokens):

| Detector | Time |
|---|---|
| heuristics | 10 s (measured, CPU) |
| fingerprint | ~2 min |
| fast_detect_gpt | ~35 min |
| binoculars | ~70 min |
| llm_judge | ~1.4 h (measured, 16-19 texts/s over the OpenWebUI endpoint) |
| detect_code_gpt | ~10 h |

A full sweep is an overnight job of roughly 12–14 hours, and DetectCodeGPT is most of
it: the method perturbs each text 50 times and rescores, so it costs 50 forward passes
where the others cost one. It is restricted to the diff genre, which is what it was
built for. Extending it to prose would be a separate decision about perturbation count,
not a free addition.

Each detector writes its own parquet, so a failure costs one detector rather than the
sweep, and the LLM judge caches to disk so re-runs are close to free.


## What actually ran, for the record

| Detector | Model | Batch | Genres | Wall clock |
|---|---|---|---|---|
| `fast_detect_gpt` | `Qwen/Qwen2.5-7B` | 16 | all | ~35 min |
| `binoculars` | `tiiuae/falcon-7b` + `-instruct` | 8 | all | ~70 min |
| `detect_code_gpt` | `codellama/CodeLlama-7b-hf` | 64 | diff only | 9 h 34 min |
| `llm_judge` | `vllm/gemma-4-31B-it` | --- | all | 1 h 33 min |

The judge ran from the workstation over the OpenWebUI endpoint rather than on the host,
so it survives the host's deletion by construction. `heuristics`, `selfadmission` and
`fingerprint` are CPU-only and run anywhere.

DetectCodeGPT was re-run after its adapter was corrected from a log-likelihood
discrepancy to the published normalised perturbed log rank. The superseded scores are
kept in `data/processed/scores_superseded/` so the correction is visible rather than
silently overwritten. Note that `host/sweep.sh` records batch size 16 for that detector;
the released scores were produced at 64, which is the figure in the table above.
