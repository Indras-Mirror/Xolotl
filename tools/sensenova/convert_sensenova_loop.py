#!/usr/bin/env python3
"""SenseNova-Loop (looped Qwen3.5 + two-stream mHC) -> GGUF for xolotl-dense.

    python3 tools/sensenova/convert_sensenova_loop.py <hf_dir> --outfile sn-loop-9b-q8_0.gguf --outtype q8_0

The text model is a Qwen3.5 body, so llama.cpp's own Qwen3_5TextModel converter does all of the trunk.  What this
subclass adds:

  * general.architecture = `qwen35loop`.  Stock llama.cpp refuses the file instead of silently running it without
    the loop; xolotl-dense admits it (every `qwen3*` arch) and reads the schedule below.
  * `qwen35loop.loop.{start,end,count}`: layers [start, end) run `count` times in sequence, each visit with its own
    KV / GDN state (the weights are shared).  SenseNova-Loop-9B: 4, 28, 2 -> 56 executions of 32 layers.
  * `qwen35loop.mhc.{start,end,streams,iterations,pre_scale,fixed_identity}` and, per mHC layer,
    `blk.N.mhc_{attn,ffn}.router` [2S + S*S, S*n_embd] and `blk.N.mhc_{attn,ffn}.bias` [2S + S*S], kept F32.
    The names do not end in `weight` on purpose: llama-quantize then leaves them alone.
  * `loop_gate.logits`, unused by the reference forward (kept for the dynamic-loop work), F32.

The vision tower is skipped here (the text filter drops `visual.*`); the mmproj is a separate pass.
Nothing in third_party/llama.cpp is modified: this file imports its conversion package and registers one class.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LLAMA = Path(__file__).resolve().parents[2] / "third_party" / "llama.cpp"
sys.path.insert(0, str(LLAMA))
sys.path.insert(1, str(LLAMA / "gguf-py"))

import torch  # noqa: E402

import gguf  # noqa: E402
import conversion  # noqa: E402
from conversion.base import ModelBase  # noqa: E402
from conversion.qwen import Qwen3_5TextModel  # noqa: E402

ARCH = "qwen35loop"
HF_ARCH = "SenseNovaLoopForConditionalGeneration"


@ModelBase.register(HF_ARCH)
class SenseNovaLoopModel(Qwen3_5TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35   # the tensor-name map of the body
    no_mtp = True                          # the release checkpoint has no mtp.* tensors

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # every key written from here on uses the writer's arch; general.architecture was already recorded by the
        # writer's constructor, so overwrite it in place (keeps it the first key)
        self.gguf_writer.arch = ARCH
        self.gguf_writer.kv_data[0][gguf.Keys.General.ARCHITECTURE] = gguf.GGUFValue(
            value=ARCH, type=gguf.GGUFValueType.STRING)
        loop = self.hparams["loop_settings"]
        if loop.get("counts") != [loop.get("eval_count")]:
            raise ValueError(f"only fixed-count loops are supported, got {loop}")
        self._mhc = self.hparams.get("mhc")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        w, loop = self.gguf_writer, self.hparams["loop_settings"]
        w.add_uint32(f"{ARCH}.loop.start", int(loop["start"]))
        w.add_uint32(f"{ARCH}.loop.end", int(loop["end"]))
        w.add_uint32(f"{ARCH}.loop.count", int(loop["eval_count"]))
        if self._mhc:
            m = self._mhc
            w.add_uint32(f"{ARCH}.mhc.start", int(m["start"]))
            w.add_uint32(f"{ARCH}.mhc.end", int(m["end"]))
            w.add_uint32(f"{ARCH}.mhc.streams", int(m["streams"]))
            w.add_uint32(f"{ARCH}.mhc.iterations", int(m["iterations"]))
            w.add_float32(f"{ARCH}.mhc.pre_scale", float(m.get("pre_scale", 1.0)))
            w.add_bool(f"{ARCH}.mhc.fixed_identity", bool(m["fixed_identity"]))

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".mhc_" in new_name or new_name.startswith("loop_gate"):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch, name, bid):
        for kind in ("mhc_attn", "mhc_ffn"):
            if f".{kind}." in name:
                assert bid is not None, name
                leaf = "router" if name.endswith(".weight") else "bias"
                yield f"blk.{bid}.{kind}.{leaf}", data_torch.to(torch.float32)
                return
        if "loop_gate" in name:
            yield "loop_gate.logits", data_torch.to(torch.float32)
            return
        yield from super().modify_tensors(data_torch, name, bid)


# convert_hf_to_gguf.main() looks the class up through this map; our module is already imported.
conversion.TEXT_MODEL_MAP[HF_ARCH] = "qwen"

if __name__ == "__main__":
    import convert_hf_to_gguf  # noqa: E402
    convert_hf_to_gguf.main()
