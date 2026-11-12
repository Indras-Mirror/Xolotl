# Credits

<img src="docs/brand/assets/logo-circle-400.png" alt="Xolotl" width="110" align="right">

## Built on

- **[Strata](https://github.com/Niko1221/Strata)** by Niko1221: its engine, server and kernels, merged at 0.1.44.1.
- **[ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)**.
- The **Qwen3.8-27B dense port** is [perronemirko](https://github.com/perronemirko/Strata)'s.
- The **DeepSeek-V4.1 and GLM-5.3-Flash engines**, the **GSQ quant**, **Xolotl Direct**, **fast mode** and the
  unification are Indras Mirror's.

## Models

- Weights: DeepSeek, Zhipu AI and Qwen, with quants and fine-tunes by peasantsmith, ukisai, d0xin and gcsa-abliterix.
- LFM3-Omni by [Liquid AI](https://huggingface.co/LiquidAI); its engine follows llama.cpp's LFM3 and libmtmd work.

## Ideas

- **Ryan Flavell** ([PROMETHEVS](https://github.com/Magnvs7/Promethevs), MAGNVS Ltd): compress-then-correct, the idea
  behind fast mode's 2-bit expert shadow and the decode margin log.

- **[NInfer](https://github.com/JGamboa/ninfer-4090-windows)** (JGamboa's RTX 4090 fork): the one-pass verify attention
  and the n-gram draft chain in `xolotl-dense`, after its `--ngram chain`; the MTP draft shortlist head
  (`--draft-vocab`, after its `--lm-head-draft`), whose token list `data/packs/qwen3.8-27b/draft-vocab-131072.i32` is
  the top 131,072 of NInfer's frequency ranking (`tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64`,
  Apache-2.0) plus the 33 control tokens; z-lab's
  [Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab) drafter for `--dflash2`.

## Contributors, with thanks

- **[NektoDron](https://github.com/NektoDron)**
  - MMQ id slack, forced tool calls, the arena pipe and indexer WMMA opt-ins (#48-#52); checked #55
  - the Q3_K + GPTQ Q2_0 lite calibration and allocation tooling (#58)
  - the ggml 6050b467 pool-hook link fix (#72); the control-queue wakeup fix (#79)
  - Qwen3.6 vision, prefix cache and batched MTP (#76); deep sleep (#75); resident prefix reuse (#80)
  - complete Unsloth GGUF and DSpark support (#84)
  - cancelled-request slot wakeups (#88); shared host checkpoint payloads (#89); learned suffix-edit checkpoints
    (#90); the bounded tokenizer span cache (#91); Qwen3.6 snapshot ownership (#92); the Metal fallback signature
    sync (#93); request-owned Flash batch cancellation (#95)
  - MiniMax M3.1 serving, Qwen picture numbering and image minimum fixes (#96, #101, #102, #104)
  - HIP shared int4 alignment (#105); completed-response and EOS-drain cancellation fixes (#107, #108)
  - the persistent prefix cache (#99); native GBNF for Qwen Flash and DeepSeek V4.1 (#112, #113)
  - nonblocking prefix lookups and the minimum restore gain (#115); the MiniMax sampling replay keyword (#116)
  - the uniform expert cache reopen after an explicit-slot release (#130); explicit CPU worker and host placement for native
    DeepSeek (#131); DeepSeek-V4.1 predictor IDs read at their row stride (#132); the upstream CUDA radix top-k cutoff-tie fix
    backport (#133)
- **[Xiaomi2008](https://github.com/Xiaomi2008)**: IQ1_S routed experts for Unsloth's DeepSeek-V4.1 UD-IQ1_M (#86); ds41
  fused router ops and batched state copies (#87); DSML tool-call fixes and the opt-in keep_cut_tool_calls (#106,
  #109, #110)
- **[AlexeyMatskevich](https://github.com/AlexeyMatskevich)**: the upstream deepseek41 GGUF layout (#82); OpenZFS ARC
  accounting (#83); IQ3_S expert downs for Unsloth's UD-Q2_K_XL (#85)
- **[paul123637-lab](https://github.com/paul123637-lab)**: the DSpark seed fixes (#54, #56); RTX 5080 and 5060 Ti helper
  numbers
- **[kelheor](https://github.com/kelheor)**: the vision `pread` fix (#51); the Windows ds41 build fixes (#94, #118: portable mkdir / 64-bit stat / free space for the dense cache and --build-shadow)
- **[exye137](https://github.com/exye137)**: the thread-local CUDA stream leak (#57)
- **[trojy13](https://github.com/trojy13)**: the Windows CRLF patch failure, its root cause and the workaround (#134)
- **[Yoongger](https://github.com/Yoongger)**: the NVIDIA L20 (92-SM Ada) interleaved-row mmvq tables and their A/B on the L20 (#135)
- **[penink-WH](https://github.com/penink-WH)**: the AC922 rig
- **[cpacsm75](https://github.com/cpacsm75)**: 192 GB arena tuning on Windows (#77); the RTX 5090 arena and `--kv-q8`
  numbers at 256K/512K (#114)
- **[jesdga95](https://github.com/jesdga95/basalt)** (Basalt): the priced stream-all prefill chunk, quoted tool calls in
  string values, union/nullable tool parameters, IQ code bytes by shift, IQ1_M on MMQ, and the ideas behind the
  DeepSeek-V4.1 Engram read-ahead and late shared expert

---

*Xolotl* is Quetzalcoatl's dark twin, who carries the dead up from the underworld - as the engine carries experts up
from RAM and NVMe.
