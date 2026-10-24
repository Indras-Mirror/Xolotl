"""analyze.py <trace.csv> <window_s> - the last window_s seconds of GPU activity: GPU busy vs idle, kernels by name."""
import collections, re, sys

path, win = sys.argv[1], float(sys.argv[2])
K, M = [], []
with open(path, encoding="utf-8", errors="replace") as f:
    for line in f:
        p = line.rstrip("\n").split(",", 4)
        if p[0] == "K":
            K.append((int(p[1]), int(p[2]), int(p[3]), p[4]))
        elif p[0] == "M":
            q = line.rstrip("\n").split(",")
            M.append((int(q[1]), int(q[2]), int(q[3]), int(q[4]), int(q[5])))
t_end = max(e for _, e, _, _ in K)
t0 = t_end - int(win * 1e9)
K = [k for k in K if k[0] >= t0]
M = [m for m in M if m[0] >= t0]
span = (t_end - t0) / 1e6

def union(iv):
    iv = sorted(iv); tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None: tot += ce - cs
    return tot / 1e6

busy = union([(s, e) for s, e, _, _ in K])
mbusy = union([(s, e) for s, e, _, _, _ in M])
print(f"window {span:.0f} ms: {len(K)} kernels, GPU kernel-busy {busy:.0f} ms ({100*busy/span:.0f}%), "
      f"memcpy-busy {mbusy:.0f} ms, {len(M)} copies ({sum(m[3] for m in M)/1e9:.2f} GB)")

def short(n):
    n = re.sub(r"<.*", "", n)            # template args
    m = re.match(r"_Z\d+(\w+?)(?:P|i|f|v|S|l|j|b|R|N|K|E|$)", n)
    if n.startswith("_Z"):
        m = re.match(r"_Z(?:N\d+\w+?)?(\d+)", n)
        if m:
            L = int(m.group(1)); i = m.end(1)
            return n[i:i + L]
    return n[:60]

agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
for s, e, st, n in K:
    a = agg[short(n)]; a[0] += 1; a[1] += (e - s) / 1e6
tot_k = sum(a[1] for a in agg.values())
print(f"sum of kernel durations {tot_k:.0f} ms; per kernel mean {1000*tot_k/len(K):.1f} us")
print(f"{'kernel':48} {'count':>7} {'ms':>8} {'%':>5} {'us/each':>8}")
for name, (c, ms, _) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:40]:
    print(f"{name[:48]:48} {c:7d} {ms:8.1f} {100*ms/tot_k:5.1f} {1000*ms/c:8.1f}")
small = [(e - s) / 1e3 for s, e, _, _ in K]
for lim in (2, 3, 5, 10):
    sel = [d for d in small if d < lim]
    print(f"kernels < {lim:2d} us: {len(sel):7d} ({100*len(sel)/len(K):.0f}%), {sum(sel)/1e3:.1f} ms")
# gaps between consecutive kernels (by start) on the busiest stream
by = collections.Counter(st for _, _, st, _ in K)
st0 = by.most_common(1)[0][0]
ks = sorted((s, e) for s, e, st, _ in K if st == st0)
gaps = [ks[i + 1][0] - ks[i][1] for i in range(len(ks) - 1)]
g = sorted(gaps)
print(f"stream {st0}: {len(ks)} kernels, gaps median {g[len(g)//2]/1e3:.1f} us, p90 {g[int(len(g)*.9)]/1e3:.1f} us, "
      f"sum {sum(x for x in gaps if x > 0)/1e6:.0f} ms (of which > 50 us: {sum(x for x in gaps if x > 50000)/1e6:.0f} ms in "
      f"{sum(1 for x in gaps if x > 50000)} gaps)")

# copies by kind and size class
kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
ca = collections.defaultdict(lambda: [0, 0, 0.0])
for s, e, st, b, k in M:
    sz = "<64K" if b < 65536 else "<1M" if b < 1 << 20 else "<16M" if b < 16 << 20 else ">=16M"
    a = ca[(kinds.get(k, str(k)), sz, st)]; a[0] += 1; a[1] += b; a[2] += (e - s) / 1e6
print("copies (kind, size, stream): count, MB, busy ms")
for key, (c, b, ms) in sorted(ca.items(), key=lambda kv: -kv[1][2])[:12]:
    print(f"  {key}: {c:6d} {b/1e6:9.0f} MB {ms:8.1f} ms")
# the big gaps on the main stream: what ran just before / after
ksn = sorted(((s, e, short(n)) for s, e, st, n in K if st == st0))
pair = collections.defaultdict(lambda: [0, 0.0])
for i in range(len(ksn) - 1):
    gp_ = ksn[i + 1][0] - ksn[i][1]
    if gp_ > 50000:
        a = pair[(ksn[i][2], ksn[i + 1][2])]; a[0] += 1; a[1] += gp_ / 1e6
print("big gaps (> 50 us) on the main stream by (kernel before -> kernel after): count, ms")
for key, (c, ms) in sorted(pair.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f"  {key[0][:30]:30} -> {key[1][:30]:30} {c:6d} {ms:8.1f}")
