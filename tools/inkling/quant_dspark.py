#!/usr/bin/env python3
"""Quantise the DSpark drafter GGUF's big matrices (FFN + attention projections) to Q8_0; everything else is copied.
The drafter only proposes tokens (the target verifies every one), so this moves acceptance, never the output.
Usage: quant_dspark.py in.gguf out.gguf"""
import re, sys
import numpy as np
sys.path.insert(0, '/home/mal/AI/Xolotl/third_party/llama.cpp/gguf-py')
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q, GGUFValueType as VT

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
arch = r.fields['general.architecture'].contents()
w = GGUFWriter(dst, arch, use_temp_file=False)
for name, f in r.fields.items():
    if name.startswith('GGUF.') or name == 'general.architecture':
        continue
    vt = f.types[0]
    if vt == VT.ARRAY:
        w.add_key_value(name, f.contents(), vt, sub_type=f.types[-1])
    else:
        w.add_key_value(name, f.contents(), vt)
pat = re.compile(r'blk\.\d+\.(ffn_(down|gate|up)|attn_(q|output|k|v)(_ctx)?)\.weight$')
saved = 0
for t in r.tensors:
    shape = [int(x) for x in reversed(t.shape)]   # numpy order (rows, cols)
    if t.tensor_type == Q.BF16:
        f32 = (np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    else:
        f32 = None
    if f32 is not None and pat.search(t.name) and shape[-1] % 32 == 0:
        q = gguf.quants.quantize(f32, Q.Q8_0)
        w.add_tensor(t.name, q, raw_shape=q.shape, raw_dtype=Q.Q8_0)
        saved += int(t.n_bytes) - q.nbytes
        print(f'q8_0 {t.name} {shape}')
    else:
        w.add_tensor(t.name, np.asarray(t.data), raw_shape=np.asarray(t.data).shape, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f'saved {saved >> 20} MiB')
