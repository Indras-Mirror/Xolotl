# Adamant

[简体中文](README.zh-CN.md)

Adamant is a fork of
[Basalt](https://github.com/jesdga95/basalt), the local inference engine for the
**Qwen3.8-Flash-Next** mixture-of-experts model, itself based on Strata. It targets NVIDIA Ada
Lovelace (RTX 40-series, sm_89) and Ampere (RTX 30-series, sm_86) GPUs; the Ampere port builds
but is not yet validated.

> **Status: early development.** The Ada port builds and passes its Linux test
> suite on an RTX 4090, but numerical quality and headline performance are still
> open gates — see [Ada execution and validation](docs/precision/ada.md). The
> performance numbers below are development runs on one card, not the
> maintainer's headline numbers. Expect breakage; flags, defaults and the pack
> layout can still change. Linux and Windows platform support is retained as a
> port requirement.

Experts that do not fit in VRAM live in system RAM, and optionally on a second GPU used as a peer expert tier.
A large MoE model runs on one workstation, behind an OpenAI- and Anthropic-compatible API.

<p align="center"><img src="docs/img/frontend.gif" alt="The web UI while a reply streams" width="720"></p>

**Help wanted:** have some spare compute to lend for testing? See [Contributing](CONTRIBUTING.md#other-ways-to-help).

## Requirements

- Linux x86-64 with AVX2 (AVX-512 is used when present), or Windows 10 1903+ x64 ([platforms](docs/platforms.md)).
- An NVIDIA Ada Lovelace (compute capability 8.9) or Ampere (8.6) GPU with CUDA 13. Ampere is not
  yet validated: the sm_86 build compiles, but parity, quality and performance on Ampere are still
  to be measured.
- CMake 3.24 or newer, GCC 13 or newer (Linux), Python 3.10 or newer, and enough system RAM for the experts that do not fit in VRAM.

## Quick start

```sh
git clone <adamant remote> && cd adamant
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j                  # bin/basalt, bin/basalt-bench, bin/basalt-probe, bin/basalt-vision
pip install -r requirements.txt
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt --local-dir models
```

The build targets sm_89 by default and also builds for sm_86; `-DCMAKE_CUDA_ARCHITECTURES`
accepts `86`, `89` or `86;89`, and anything else is refused. The toolchain and test invocations
used for the results below are in [Ada execution and validation](docs/precision/ada.md), which
also lists what is still open on Ampere.

The binary and file-format names stay `basalt` and `.basalt`: the packs are the
same files Basalt reads, so a fork that renamed them would break every existing
model. Adamant is the project name, not a new format.

Put your settings in `basalt_config.json`: the `models` list names the packs, and each entry's own `gpu` and `kv` win over the
top-level `engine` block ([models and the router](docs/configuration/models.md)):

```json
{
  "version": 2,
  "models": [{ "model": "models/Qwen3.8-FN-GSQ-RCO-IQ3_S.basalt", "gpu": "0,1", "kv": "int8" }],
  "engine": { "expert-cache": "auto", "prefill": "auto", "resident-experts": true },
  "server": { "port": 8080 }
}
```

```sh
bin/basalt serve --config basalt_config.json
```

The server listens on 127.0.0.1. Every setting: [configuration](docs/configuration/index.md). More examples:
[example configurations](docs/configuration/examples.md).

## Measured on Ada

Development measurements, not the maintainer's headline runs: one RTX 4090 at a
**450 W** power limit, the GSQ-RCO IQ3_S pack converted with `tools/convert`,
int8 KV, expert cache auto — all experts held in 17.4 GiB of VRAM and 29.4 GiB of
page-locked RAM, none read from the SSD.

A measured Adamant-versus-Strata development comparison across three fixtures
and two prompt lengths is documented in [the comparison report](docs/benchmarks/strata-comparison.md).

`basalt-bench` fixtures, MTP drafter off → on: conv 115 → 190, struct 116 → 372,
prose 116 → 169 tok/s (draft acceptance 69%, 94%, 60%). The spread is drafting
acceptance per content, not engine speed. No accuracy-reducing option is enabled
(`--route-tail-skip`, `--route-prior`, `--relaxed-drafting` are off).

## Models

| | Qwen3.8-Flash-Next |
|---|---|
| Packs | default `Qwen3.8-FN-NVFP4-DQ6_K`, [Hugging Face](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) |
| Context | default 8,192; `--max-context` sets it |
| Drafting | MTP drafter in the pack, on; `--spec` default 8 |
| Images | yes |
| Guide | [guide](docs/models/qwen38-flash-next.md) |
| Precision | [precision](docs/precision/qwen38-flash-next.md) |

All models compared: [models](docs/models/index.md).

On Ada the NVFP4 packs run through the integer MMQ path rather than native FP4
tensor instructions, so their numerics differ from a Blackwell run; the IQ3_S,
Q4_K and Q5_K formats use the same kernels. `Qwen3.8-FN-GSQ-RCO-IQ3_S` is the
quant the measurements above used, and the recommended starting point on a
24 GB card. A pack can be built locally from a GGUF, drafter included
([model file](docs/model-file.md), [precision on Ada](docs/precision/ada.md)).

## Documentation

- **Run:** [running](docs/running.md), [configuration](docs/configuration/index.md), [models and the router](docs/configuration/models.md),
  [sampling](docs/configuration/sampling.md), [image sources](docs/configuration/image-sources.md),
  [diagnostics](docs/configuration/diagnostics.md), [basalt-probe](docs/probe.md), [platforms](docs/platforms.md).
- **Models:** [models](docs/models/index.md), [Flash-Next](docs/models/qwen38-flash-next.md),
  [the `.basalt` file](docs/model-file.md), [file format changes](docs/basalt-file-changelog.md).
- **Quality:** [precision](docs/precision/index.md), [method](docs/precision/quality.md).
- **Build and develop:** [building](docs/building.md), [adding a model](docs/adding-a-model.md), [community runs](docs/community.md),
  [changelog](docs/changelog/index.md), [contributing](CONTRIBUTING.md).

## Credits

- [Basalt](https://github.com/jesdga95/basalt): the project Adamant forks (MIT).
- [Strata](https://github.com/Niko1221/Strata): the project Basalt forks (MIT).
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): the model. The packs are under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), not Basalt's MIT license.
- [unsloth](https://huggingface.co/unsloth): the UD and Q8_0 GGUFs the packs are built from.
- IST Austria DAS Lab: the GSQ-RCO quants; please cite [GSQ](https://arxiv.org/abs/2604.18556) and [RCO](https://arxiv.org/abs/2605.00649).
- [NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4): the NVFP4 experts.
- [ggml and llama.cpp](https://github.com/ggml-org/llama.cpp): quantization formats and kernels extracted into `src/` (MIT, see [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES)).

## License

MIT, see [LICENSE](LICENSE).
