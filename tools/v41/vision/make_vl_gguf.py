r"""Add DeepSeek V4.1's image-position router bias (blk.N.exp_probs_b_vl.bias) to the GSQ-RCO IQ2_XS split GGUF.

ISTA-DASLab's llama.cpp-convention GGUF leaves the 40 bias_vl tensors out, so Xolotl refuses --vision.  Its 40
exp_probs_b.bias tensors are bitwise identical to Unsloth's (the checkpoint's own F32 values), so the checkpoint's
bias_vl, taken from the converted Unsloth GGUF, belongs to the same router.

Output: a NEW shard family in --out-dir.  Shard 1 is rewritten: the same metadata minus split.tensors.count (the
engine then skips the per-shard count check, which the unchanged shards 2-4 would fail), the same tensor directory
plus the 40 new entries, the same data section byte for byte, then the new tensors.  Shards 2-4 are hard links to
the originals (no copy, no change).  The source files are only read.  Every byte of the old data region is verified.
The 40 tensors ship with this repo as data/ds41/bias_vl-1017.gguf (taken from a converted Unsloth GGUF and checked
equal to it), so no Unsloth download is needed:

  python make_vl_gguf.py --src-shard1 D:\models\dsv41-1017-gsq-iq2xs\DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf ^
      --bias-from data\ds41\bias_vl-1017.gguf --out-dir D:\models\dsv41-1017-gsq-iq2xs-vl ^
      --out-prefix DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-vl

--out-dir must be a new folder on the same drive as the source (hard links).  It needs numpy and the gguf package
(the repo's third_party/llama.cpp/gguf-py after tools/xolotl/fetch-llama.sh, or pip install gguf).
"""
import argparse
import hashlib
import os
import struct
import sys
from pathlib import Path

GGUF_TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # 8 string, 9 array
F32 = 0


class Header:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            head = f.read(64 << 20)
        self.raw = head
        p = 0

        def take(n):
            nonlocal p
            b = head[p:p + n]
            if len(b) != n:
                raise ValueError('header larger than the read window')
            p += n
            return b

        def u32():
            return struct.unpack('<I', take(4))[0]

        def u64():
            return struct.unpack('<Q', take(8))[0]

        def string():
            return take(u64()).decode('utf-8')

        def skip_value(t):
            if t == 8:
                take(u64())
            elif t == 9:
                et, n = u32(), u64()
                for _ in range(n):
                    skip_value(et)
            else:
                take(GGUF_TYPE_SIZES[t])

        if take(4) != b'GGUF':
            raise ValueError('not a GGUF file')
        self.version = u32()
        n_tensors, n_kv = u64(), u64()
        self.kvs = []                       # (key, raw bytes of the whole KV entry, value for scalars)
        self.alignment = 32
        for _ in range(n_kv):
            start = p
            key = string()
            t = u32()
            vstart = p
            skip_value(t)
            val = None
            if t in (4, 5) and p - vstart == 4:
                val = struct.unpack('<I' if t == 4 else '<i', head[vstart:p])[0]
            elif t in (10, 11):
                val = struct.unpack('<Q' if t == 10 else '<q', head[vstart:p])[0]
            elif t == 2 or t == 3:
                val = struct.unpack('<H' if t == 2 else '<h', head[vstart:p])[0]
            self.kvs.append((key, head[start:p], val))
            if key == 'general.alignment':
                self.alignment = val
        self.tensors = []                   # (name, dims, type, offset, raw directory entry)
        for _ in range(n_tensors):
            start = p
            name = string()
            nd = u32()
            dims = [u64() for _ in range(nd)]
            ty = u32()
            off = u64()
            self.tensors.append((name, dims, ty, off, head[start:p]))
        self.info_end = p
        self.data_start = (p + self.alignment - 1) // self.alignment * self.alignment
        self.file_size = os.path.getsize(path)

    def kv(self, key):
        for k, _, v in self.kvs:
            if k == key:
                return v
        return None


def align(n, a):
    return (n + a - 1) // a * a


def tensor_entry(name, dims, ty, off):
    nb = name.encode('utf-8')
    return (struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(dims)) +
            b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<I', ty) + struct.pack('<Q', off))


def read_bias_vl(path, n_layers):
    vendored = Path(__file__).resolve().parents[3] / 'third_party' / 'llama.cpp' / 'gguf-py'
    if vendored.is_dir():
        sys.path.insert(0, str(vendored))
    from gguf import GGUFReader
    import numpy as np
    out = {}
    for t in GGUFReader(path).tensors:
        if t.name.endswith('.exp_probs_b_vl.bias'):
            if t.tensor_type.name != 'F32':
                raise ValueError(f'{t.name}: {t.tensor_type.name}, expected F32')
            out[t.name] = (list(int(x) for x in t.shape), np.array(t.data, dtype=np.float32).tobytes())
    want = [f'blk.{i}.exp_probs_b_vl.bias' for i in range(n_layers)]
    if sorted(out) != sorted(want):
        raise ValueError(f'{path}: expected {n_layers} bias_vl tensors, found {len(out)}')
    return [(n, *out[n]) for n in want]


def sha256_range(path, start, length, chunk=64 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        f.seek(start)
        left = length
        while left:
            b = f.read(min(chunk, left))
            if not b:
                raise IOError(f'{path}: short read')
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src-shard1', required=True)
    ap.add_argument('--bias-from', required=True, help='GGUF holding blk.N.exp_probs_b_vl.bias (F32)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--out-prefix', required=True, help='e.g. DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-vl')
    a = ap.parse_args()

    src = Path(a.src_shard1)
    import re
    m = re.match(r'(.+)-00001-of-(\d{5})\.gguf$', src.name)
    if not m:
        raise SystemExit('--src-shard1 must be a -00001-of-NNNNN.gguf file')
    n_shards = int(m[2])
    H = Header(str(src))
    n_layers = H.kv('deepseek41.block_count')
    if H.kv('split.no') != 0 or H.kv('split.count') != n_shards:
        raise SystemExit('source shard 1 split keys do not match its name')
    if any(t[0].endswith('exp_probs_b_vl.bias') for t in H.tensors):
        raise SystemExit('source already has exp_probs_b_vl.bias')
    biases = read_bias_vl(a.bias_from, n_layers)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    out1 = out_dir / f'{a.out_prefix}-00001-of-{n_shards:05d}.gguf'
    old_data_len = H.file_size - H.data_start
    base = align(old_data_len, H.alignment)
    kv_raw = [raw for k, raw, _ in H.kvs if k != 'split.tensors.count']
    entries = [t[4] for t in H.tensors]
    new_off = []
    off = base
    for name, dims, data in biases:
        new_off.append(off)
        entries.append(tensor_entry(name, dims, F32, off))
        off = align(off + len(data), H.alignment)
    header = (b'GGUF' + struct.pack('<I', H.version) + struct.pack('<Q', len(entries)) +
              struct.pack('<Q', len(kv_raw)) + b''.join(kv_raw) + b''.join(entries))
    new_data_start = align(len(header), H.alignment)
    with open(src, 'rb') as fi, open(out1, 'wb') as fo:
        fo.write(header + b'\0' * (new_data_start - len(header)))
        fi.seek(H.data_start)
        left = old_data_len
        while left:
            b = fi.read(min(256 << 20, left))
            if not b:
                raise IOError('short read of the source data section')
            fo.write(b)
            left -= len(b)
        fo.write(b'\0' * (base - old_data_len))
        for (name, dims, data), o in zip(biases, new_off):
            pos = fo.tell() - new_data_start
            if pos != o:
                raise AssertionError(f'{name}: at {pos}, directory says {o}')
            fo.write(data + b'\0' * (align(len(data), H.alignment) - len(data)))
    # links for the unchanged shards
    for i in range(2, n_shards + 1):
        s = src.with_name(f'{m[1]}-{i:05d}-of-{m[2]}.gguf')
        os.link(s, out_dir / f'{a.out_prefix}-{i:05d}-of-{n_shards:05d}.gguf')

    # verification: directory, data bytes, appended values
    N = Header(str(out1))
    assert N.alignment == H.alignment and N.data_start == new_data_start
    assert N.kv('split.tensors.count') is None and N.kv('split.count') == n_shards and N.kv('split.no') == 0
    assert [k for k, _, _ in N.kvs] == [k for k, _, _ in H.kvs if k != 'split.tensors.count']
    assert [t[:4] for t in N.tensors[:len(H.tensors)]] == [t[:4] for t in H.tensors]
    old_h = sha256_range(str(src), H.data_start, old_data_len)
    new_h = sha256_range(str(out1), N.data_start, old_data_len)
    assert old_h == new_h, 'data section differs'
    with open(out1, 'rb') as f:
        for (name, dims, data), o in zip(biases, new_off):
            f.seek(N.data_start + o)
            assert f.read(len(data)) == data, name
    print(f'OK {out1.name}: {len(H.tensors)} + {len(biases)} tensors, data sha256 {old_h[:16]} unchanged, '
          f'{os.path.getsize(out1)} bytes; shards 2-{n_shards} hard-linked')


if __name__ == '__main__':
    main()
