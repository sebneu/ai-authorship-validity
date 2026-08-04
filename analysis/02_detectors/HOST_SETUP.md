# GPU host requirements

What the detector runs need from the VM, with the reasoning, so the sysadmin can size
it without a back-and-forth.

## GPU

**RTX PRO 6000 Blackwell (96 GB) is fine, and slightly better than the H100 PCIe here.**
The workload is inference prefill on 7B models, which is compute- and
bandwidth-bound rather than capacity-bound. 96 GB is more VRAM than an H100 PCIe (94 GB),
and holds the two models Binoculars needs simultaneously with room to spare. No NVLink,
FP64 or multi-GPU capability is required: everything runs on one card.

**One hard constraint.** Blackwell is compute capability 12.0 (`sm_120`). PyTorch wheels
built for CUDA 12.4 or earlier contain no `sm_120` kernels and fail at model load with
*"no kernel image is available for execution on the device"*. The host needs:

- NVIDIA driver **570 or newer**
- **CUDA 12.8**
- **PyTorch 2.7+** built for cu128 (pinned in `requirements-gpu.txt`)

This is the one setting that silently wastes a day if it is wrong.

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
