# DeepSeek V4.1 upstream GGUF assets

The engine accepts both its original GGUF convention and the upstream llama.cpp
`deepseek41` convention. This extends the latter with the complete Unsloth
DeepSeek-V4.1-Flash-1017 asset set, rather than substituting Engram rows from a
different quantization or checkpoint.

## Engram rows and hash metadata

Upstream GGUFs may contain one `per_layer_token_embd.weight` lookup tensor. The
lazy Engram reader accepts Q5_K and IQ4_NL rows, concatenated by Engram layer and
then by hash bucket. It reads only requested rows through the existing row cache,
coalesced reads and asynchronous workers. It does not allocate the entire table
in the dense GPU arena.

Pass the shard containing that tensor as an existing `--engram` argument. A
separate compatible Engram GGUF or the original safetensors tables remain valid.
A header-only final backbone shard remains sufficient when an external Engram
source is supplied; opening the lookup source itself requires its complete
payload. No lookup bytes are inferred or replaced from another model.

`ple.head_offsets` and `engram.compressed_vocab_size` remain optional. When
present, offsets must match the exact layer-major running sum. The token map,
layers, bucket sizes and multipliers are checked before narrowing or hashing;
the pad tokenizer ID is translated through that same compressed token map.
The grouped attention output accepts its exact 2-D or 3-D contiguous view.
Optional vision-router bias remains optional.

Folded `engram_gate` weights retain the loader's conversion to F32 for every
source type. Their graph follows the reference multiply order and signed-square-
root convention, including zero; legacy separate q/k weights retain their
original arithmetic.

## DSpark

The existing `STRATA_DSPARK` sidecar option also accepts upstream `dflash` GGUFs.
The importer validates target layers, stage count, tensor shapes and types before
mapping the DFLASH names to the existing DSpark graph. Legacy `mtp.*` sidecars
keep their original loader. Dense BF16/Q8_0 operands are converted as required;
routed experts keep their native quantized bytes.

The encoder's existing vision asset format and serving APIs are unchanged. A
GGUF vision projector is not imported by this change; supply a compatible encoder
through the existing vision configuration.

## Expert kernels and checks

IQ3_S down-projections now use the existing native IQ3_S dot-product kernel.
Gate/up and down role availability share one CUDA-free registry with header
admission. The CPU checker reports unsupported role combinations before GPU
startup; this is not a numerical GPU test.

CPU targets: `test_ds41_ple`, `test_ds41_dspark_gguf`,
`test_native_expert_registry`, and `check_ds41_gguf_headers <first-shard.gguf>`.
The focused GPU target `native_expert_iq3s_down_parity --synthetic
 iq2_s/iq3_s iq3_xxs/iq3_s` uses width 5120 and FF 2304 with the existing parity
references and tolerances. `native_grouped_parity` also covers both new down
combinations. GPU targets must run on an available device.

These tests cover metadata admission, row/hash mapping, malformed inputs,
sidecar mapping and native kernel parity. End-to-end quality, speculative
acceptance, cache, vision and sleep still require model-level validation on the
exact combined revision.
