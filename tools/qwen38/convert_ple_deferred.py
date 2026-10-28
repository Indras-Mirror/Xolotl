#!/usr/bin/env python3
"""Convert Qwen3.8-Flash-Next to GGUF with the PLE n-gram table DEFERRED.

Why this exists
---------------
The PLE table (`per_layer_token_embd`, 51.2B params) is F8_E4M3 in the checkpoint, split into 128
`ngram_embedding.shard_N.weight` tensors living in 31 dedicated shards (49.6 GB).  We do not
download those, because:

  * the engine only ever reads this table as IQ4_NL (ngram.cpp), so we supply it at serve time
    from a ready-made IQ4_NL GGUF via --ple-gguf;
  * letting the converter assemble the table would materialise it as float32 - roughly 205 GB -
    inside the output GGUF, which is both huge and a different format from what we serve.

The upstream converter is *almost* fine with that: `_place_ple_shard` returns [] until it has all
`split_ngram_parts` shards, so a partial set simply produces no table tensor.  The one blocker is
`prepare_tensors`, which raises "got N PLE embedding shards, expected 128" on a partial set.

Rather than edit the vendored llama.cpp tree (this branch carries local qwen4exp work we do not
want to conflict with), this wrapper imports the model class, replaces that single method with a
version that tolerates a partial set when QWEN_PLE_DEFERRED=1, and then runs the normal converter.
The default behaviour of the converter is unchanged; without the env var this wrapper is a no-op.

What is still emitted: the PLE hash constants and the row width come from the mixed shards we DO
download (model-00005 carries layer_multipliers / key_proj / norms, model-00037 carries
ngram_heads_offsets / ngram_heads_vocab_sizes / value_proj), so the metadata keys the engine needs
are still written.  This is the same split tools/nvfp4_convert.py documents.

  QWEN_PLE_DEFERRED=1 python3 tools/qwen38/convert_ple_deferred.py \
      "/media/mal/Team CX2/models/Qwen3.8-Flash-Next-FP8" \
      --outfile /path/model-Q8_0.gguf --outtype q8_0
"""
import os
import pathlib
import runpy
import sys

LLAMA_DIR = pathlib.Path(os.environ.get(
    "LLAMA_QWEN4EXP_DIR", "/home/mal/AI/llama.cpp-qwen4exp")).expanduser()

if str(LLAMA_DIR) not in sys.path:
    sys.path.insert(0, str(LLAMA_DIR))

import conversion.qwen4exp as q4  # noqa: E402  (registers the class in ModelBase)

if os.environ.get("QWEN_PLE_DEFERRED") != "1":
    print("convert_ple_deferred: QWEN_PLE_DEFERRED is not 1 - the PLE table will be expected.\n"
          "  Set QWEN_PLE_DEFERRED=1 to convert without the 31 PLE table shards.", file=sys.stderr)
else:
    _orig_prepare = q4.Qwen4ExpTextModel.prepare_tensors

    def prepare_tensors(self):
        try:
            _orig_prepare(self)
        except ValueError as e:
            # the only error we are allowed to swallow: a PARTIAL set of PLE table shards
            if "PLE embedding shards" not in str(e):
                raise
            print(f"convert_ple_deferred: deferring the PLE table ({e})\n"
                  f"  the table is supplied at serve time via --ple-gguf", file=sys.stderr)
            # _place_ple_shard already recorded _ple_row_dim from a shard it did see, so the
            # metadata still carries embedding_length_per_layer_input (160 for this model).

    if q4.Qwen4ExpTextModel.prepare_tensors is not _orig_prepare:  # pragma: no cover
        raise SystemExit("convert_ple_deferred: prepare_tensors was already patched")
    q4.Qwen4ExpTextModel.prepare_tensors = prepare_tensors

    # Second breakage: the table's per-tensor FP8 scale.
    # `ngram_embedding.weight_scale` is BF16 [1] - one scalar for the whole 51.2B table, not the
    # 128x128 `weight_scale_inv` blocks the fp8 branch routes (base.py:431 only pairs _scale_inv,
    # and the `.weight_scale` branch at base.py:550 is gated on quant_method == "modelopt", while
    # this checkpoint is quant_method "fp8").  So it falls through to map_tensor_name and raises
    # "Can not map tensor".  Dropping it is correct here and only here: it scales a table that is
    # never emitted on this path.  Note the latent upstream bug this hides - _place_ple_shard
    # casts the FP8 shards to f32 WITHOUT applying this scale, so a full 128-shard conversion
    # would emit an unscaled table.  Deferring the table sidesteps that entirely.
    _orig_modify = q4.Qwen4ExpTextModel.modify_tensors

    def modify_tensors(self, data_torch, name, bid):
        if name.endswith(".ngram_embedding.weight_scale"):
            print(f"convert_ple_deferred: dropping {name} (scale for the deferred table)",
                  file=sys.stderr)
            return []
        return _orig_modify(self, data_torch, name, bid)

    if q4.Qwen4ExpTextModel.modify_tensors is not _orig_modify:  # pragma: no cover
        raise SystemExit("convert_ple_deferred: modify_tensors was already patched")
    q4.Qwen4ExpTextModel.modify_tensors = modify_tensors

runpy.run_path(str(LLAMA_DIR / "convert_hf_to_gguf.py"), run_name="__main__")
