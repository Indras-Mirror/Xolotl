#!/usr/bin/env python3
"""extract_engram.py SRC_DIR [SRC_DIR2 ...] DST_DIR
   or: extract_engram.py SRC_DIR ... --to layers.1=DIR_A --to layers.14=DIR_B   (one file per layer, per drive)

Copies only the Engram hash tables of DeepSeek-V4.1-Flash (layers.<L>.engram.embed.weight / .scale for every
Engram layer, ~202 GB) out of the HF safetensors shards into ONE new safetensors file in DST_DIR, plus a
model.safetensors.index.json that maps them to it.  Then point the engine at it:  --engram DST_DIR

Why: the engine reads ~48 random 256-byte Engram rows per decoded token.  On an exFAT volume random preads
serialize (~13K reads/s measured on NVME1TB) - ~5-10 ms per token; on the ntfs3 / ext4 NVMe the same reads run at
~290K reads/s.  Only these four tensors are read at run time (ds41_engram.cpp Ds41Engram::open).

Checks free space first, streams with 64 MiB buffers, verifies every tensor's size and (with --verify N) N random
rows against the source.  Safe to re-run: writes DST_DIR/engram.safetensors.partial and renames when complete.
"""
import json, os, random, shutil, struct, sys, time

def st_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n

def main():
    argv = sys.argv[1:]
    verify, routes, args = 2000, {}, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('--verify='): verify = int(a.split('=', 1)[1])
        elif a == '--to': k, v = argv[i + 1].split('=', 1); routes[k] = v; i += 1
        else: args.append(a)
        i += 1
    if routes:
        if len(args) < 1: print(__doc__); sys.exit(2)
        srcs = args
    else:
        if len(args) < 2: print(__doc__); sys.exit(2)
        srcs = args[:-1]
    index = None
    for d in srcs:
        p = os.path.join(d, 'model.safetensors.index.json')
        if os.path.exists(p):
            index = json.load(open(p))['weight_map']; break
    if index is None: sys.exit('no model.safetensors.index.json in ' + ' '.join(srcs))
    names = sorted(k for k in index if '.engram.embed.' in k)
    if not names: sys.exit('no engram.embed tensors in the index')
    def find(fn):
        for d in srcs:
            if os.path.exists(os.path.join(d, fn)): return os.path.join(d, fn)
        sys.exit('shard not found: ' + fn)
    plan = []   # (name, src path, src abs offset, nbytes, dtype, shape)
    for nm in names:
        path = find(index[nm])
        h, ds = st_header(path)
        t = h[nm]
        a, b = t['data_offsets']
        plan.append((nm, path, ds + a, b - a, t['dtype'], t['shape']))
    if routes:   # layers.<L>.engram.embed.* -> (dir, engram-<L>.safetensors)
        groups = {}
        for p in plan:
            pre = p[0].split('.engram.')[0]
            if pre not in routes: sys.exit(f'no --to for {pre}')
            groups.setdefault((routes[pre], 'engram-' + pre.replace('layers.', 'L') + '.safetensors'), []).append(p)
    else:
        groups = {(args[-1], 'engram.safetensors'): plan}
    filemap = {p[0]: fn for (d, fn), ps in groups.items() for p in ps}
    for (dst, fn), ps in groups.items():
        write_one(dst, fn, ps, verify)
    for d in sorted({d for d, _ in groups}):
        json.dump({'metadata': {'total_size': sum(p[3] for p in plan)}, 'weight_map': filemap},
                  open(os.path.join(d, 'model.safetensors.index.json'), 'w'), indent=1)
    print('use: --engram "' + ':'.join(sorted({d for d, _ in groups})) + '"')

def write_one(dst, fname, plan, verify):
    total = sum(p[3] for p in plan)
    os.makedirs(dst, exist_ok=True)
    free = shutil.disk_usage(dst).free   # portable (os.statvfs is POSIX-only)
    print(f'{len(plan)} tensors, {total/1e9:.1f} GB -> {dst} (free {free/1e9:.1f} GB)')
    for p in plan: print(f'  {p[0]:40s} {p[4]:8s} {p[3]/1e9:7.2f} GB  from {os.path.basename(p[1])}')
    if free < total + (1 << 30): sys.exit(f'not enough free space: need {total/1e9:.1f} GB + 1 GB headroom')
    # new header: tensors back to back, 64-byte aligned
    hdr, off = {}, 0
    for nm, _, _, nb, dt, shp in plan:
        hdr[nm] = {'dtype': dt, 'shape': shp, 'data_offsets': [off, off + nb]}
        off += nb
        off = (off + 63) & ~63
    hj = json.dumps(hdr, separators=(',', ':')).encode()
    hj += b' ' * ((8 - (8 + len(hj)) % 8) % 8)
    out = os.path.join(dst, fname)
    part = out + '.partial'
    BUF = 64 << 20
    t0 = time.time(); done = 0
    with open(part, 'wb') as fo:
        fo.write(struct.pack('<Q', len(hj))); fo.write(hj)
        base = 8 + len(hj)
        for nm, path, so, nb, *_ in plan:
            fo.seek(base + hdr[nm]['data_offsets'][0])
            with open(path, 'rb', buffering=0) as fi:
                fi.seek(so); left = nb
                while left:
                    chunk = fi.read(min(BUF, left))
                    if not chunk: sys.exit(f'short read in {path}')
                    fo.write(chunk); left -= len(chunk); done += len(chunk)
                    if done % (4 << 30) < BUF:
                        el = time.time() - t0
                        print(f'  {done/1e9:7.1f} / {total/1e9:.1f} GB  {done/1e9/max(el,1e-9):.2f} GB/s', flush=True)
        fo.truncate(base + off)
    # verify sizes + random rows
    h2, ds2 = st_header(part)
    rnd = random.Random(1)
    with open(part, 'rb') as fn_:
        for nm, path, so, nb, *_ in plan:
            a, b = h2[nm]['data_offsets']
            assert b - a == nb, nm
            row = 256 if nm.endswith('weight') else 8
            with open(path, 'rb') as fs:
                for _ in range(verify):
                    r = rnd.randrange(nb // row)
                    fs.seek(so + r * row); fn_.seek(ds2 + a + r * row)
                    if fs.read(row) != fn_.read(row): sys.exit(f'VERIFY FAILED {nm} row {r}')
    os.replace(part, out)
    print(f'{out}: done in {time.time()-t0:.0f} s, verified {verify} random rows per tensor')

if __name__ == '__main__':
    main()
