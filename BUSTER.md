# Buster: DeepSeek-V4.1-Flash-1017 on 2x Tesla V100-32GB

This page is for **buster** only: dual V100-SXM2 (sm_70), Broadwell dual-socket host RAM, fast NVMe for Engram, SATA/NVMe for the GSQ GGUF. It is not a supported matrix entry for arbitrary hardware.

## What works today (this branch)

| Area | Status |
| --- | --- |
| Build for sm_70 + sm_86 (`STRATA_EXPERIMENTAL_SM60`, CUDA 12.x) | `tools/buster/cmake-buster.sh` (`CMAKE_CUDA_ARCHITECTURES='70-real;86-real'`) |
| **Head GPU (3090) for output + DSpark** | `--head-device 0 --stage-devices 1,2` (experts/arena only on V100s) |
| **Layer split across both V100s (DS41)** | `--layer-split` + `gpu: [1,2]`; per-stage `ggml_backend_cuda_init`; PCIe handoff |
| **Conversation cache (DS41)** | `--conversation-cache-*`, multi-stage capture/restore |
| DSpark on sm_70 layer-split | Validated (ac0cbe8a); optional head offload on 3090 |

### Driver / toolkit note (buster: 550 + CUDA 12.9)

NVIDIA driver **550** officially supports up to **CUDA 12.4**. A **12.9** toolkit on that driver relies on **minor-version compatibility** (forward-compatible driver + older max supported CUDA). Volta has **no PTX JIT fallback** for missing SASS: the build must embed **native sm_70 machine code**. `cmake-buster.sh` sets `-DCMAKE_CUDA_ARCHITECTURES='70-real;86-real'` so nvcc emits cubins for V100 and an optional RTX 3090 head card, not PTX-only virtual arch.

`STRATA_SM70_TABLE`, `STRATA_PROMPT_ATTN_OLD`, and `STRATA_BF16_TC` in `tools/buster/buster-engine.sh` are **process-wide** runtime toggles in ggml (they follow the minimum compute capability among visible CUDA devices). With a 3090 visible alongside V100s, the minimum CC is still 7.0, so Volta paths stay selected for all GPUs in the process. The 3090 head only runs the output matmul, DSpark, and small state — not the heavy sm_70 attention path.

Host **GCC** must be one nvcc accepts (often 11 or 12; **gcc 13** may work on CUDA 12.9 but is not the documented default).

## Build (on buster)

```bash
git clone <this-fork> /opt/xolotl && cd /opt/xolotl
bash tools/xolotl/fetch-llama.sh
bash tools/buster/cmake-buster.sh /opt/xolotl/build-buster
ninja -C /opt/xolotl/build-buster ds41_generate test_cuda_cc_policy test_ds41_pipeline test_ds41_geometry
```

Serve is **not** a ninja target — it is the Python wrapper:

```bash
python3 tools/v41/serve/serve_ds41.py --engine strata --config tools/v41/serve/strata-buster-ds41-1017.json --port 8150
```

### ctest (registered targets)

```bash
cd /opt/xolotl/build-buster
ctest -R 'test_cuda_cc_policy|test_ds41_pipeline|test_ds41_geometry' --output-on-failure
```

`test_ds41_geometry` **skips** (exit 77) when no GGUF is given; set `DS41_TEST_GGUF=/path/to/V41-GSQ-1017.gguf` to run the full gate on buster.

`test_ds41_ops` is built by ninja but **not** registered in ctest (it needs reference `.f32` files on disk).

### `test_ds41_ops` reference files

`test_ds41_ops DIR` expects `DIR/x.f32`, `DIR/fp8.f32`, `DIR/fp4e.f32`, `DIR/fp4p.f32` (torch reference rounding). Generate them without the 1017 GGUF:

```bash
python3 tools/buster/gen_ds41_ops_fq_refs.py /tmp/ds41-fq-refs
./build-buster/test_ds41_ops /tmp/ds41-fq-refs
./build-buster/test_ds41_ops /tmp/ds41-fq-refs --cuda   # optional GPU path
```

Requires **PyTorch** (same rounding as `tools/v41/torch_kernels/kernel.py`). For bit-exact checks against a live model run, Mal's bring-up used a saved dir under `v41-dev/fq` from `tools/v41/engine/gpu_bringup.sh`.

### CPU-only sanity (any machine)

```bash
c++ -std=c++17 -I include tools/buster/test_cuda_cc_policy.cpp -o /tmp/test_cuda_cc_policy && /tmp/test_cuda_cc_policy
```

## Run

```bash
# (a) 2-stage trunk on V100s, head + DSpark on 3090:
export CUDA_VISIBLE_DEVICES=0,1,2
# Engine: --head-device 0 --stage-devices 1,2 --layer-split 20

# (b) 3-stage: early layers + head + DSpark on 3090, rest on V100s (example split 9 / 25):
# --head-device 0 --stage-devices 0,1,2 --layer-split 9,25
# Optional per-stage PCIe: --stage-pcie 0.55,0.25,0.25
# At load: handoff lines like "handoff stage 0 -> 1 (GPU 0 -> 1): pinned-host"
# STRATA_SELFTEST_HANDOFF=1 round-trips one column across every boundary (fails fast if broken)
# STRATA_BUILD_TYPE=RelWithDebInfo bash tools/buster/cmake-buster.sh ...  # gdb-friendly build
export STRATA_BUILD_DIR=/opt/xolotl/build-buster
python3 tools/v41/serve/serve_ds41.py --engine strata --config tools/v41/serve/strata-buster-ds41-1017.json --port 8150
```

`server.py` adds `--layer-split 20` from `layer_split` and sets `CUDA_VISIBLE_DEVICES` from `gpu: [1,2]` (PCI bus order).

## Validation plan (run on buster)

1. **Smoke:** `curl http://127.0.0.1:8150/health`; stderr: `ds41: layer split at layer 20 across GPUs 0 and 1` (visible indices after `CUDA_VISIBLE_DEVICES`).
2. **Dual-GPU:** `nvidia-smi` during decode — memory on **both** cards.
3. **Conversation cache:** interleaved long prompts → `strata serve: conversation cache: parked` / `restored`.
4. **Logits / PPL:** `--dump-last` or `--ppl` vs a known-good run (buster-only).
5. **Unit tests:** `ctest` line above; optional `test_ds41_ops` with generated refs.

## sm_70 kernel notes

Chunk-MMQ off below sm_75; MMVQ default; `STRATA_PROMPT_ATTN_OLD=1` and `STRATA_BF16_TC=1` via `tools/buster/buster-engine.sh`.

## Known risks

1. End-to-end dual-GPU correctness and throughput (hardware-only).
2. DSpark on sm_70 untested.
3. `--split-skip-if-fits` accepted for router JSON only (not implemented for DS41).
