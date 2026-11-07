#!/usr/bin/env python3
"""Quality scorer for our V4.1 engine against DwarfStar's official API fixtures (cases from build_cases.py).

run:   start ds41_generate --serve once (through memguard), send every case as `GEN 1 <ids>` with the continuation
       teacher-forced inside the prompt, and let the engine dump each request's last rows (STRATA_SERVE_DUMP_TAIL).
score: per position of the continuation, from the engine's logits row (row p predicts token p+1):
       NLL of the API's sampled token, top-1 agreement with the API's top token, and the overlap lower bound
       sum(min(p_local, p_api)) over the API's top-20 (DwarfStar's metrics; comparable to their 90.3% top-1 for
       IQ2_XXS-Q2_K).

usage: score_official.py run   <cases.jsonl> <outdir> [engine args...]   (env passes through: STRATA_SERVE_*, ...)
       score_official.py score <cases.jsonl> <outdir> [<outdir2> ...]
"""
import json, os, subprocess, sys, time
import numpy as np

NV = 129280
DS = os.path.expanduser('~/AI/Xolotl')
BASE_ARGS = ['-m', '/media/mal/SSD NVME/Xolotl/deepseek-v4.1/V41-GSQ/V41-GSQ.gguf', '--backend', 'cuda', '--experts', 'gpu',
             '--engram', '/media/Crucial1TB/V41-Engram:/media/mal/SSD NVME/AI/V41-Engram',
             '--rank', '/media/mal/SSD NVME/Xolotl/deepseek-v4.1/V41-GSQ/rank.bin', '--ctx', '32768', '--stop', '1', '--serve']


def load_cases(path):
    return [json.loads(l) for l in open(path)]


def run(cases_path, out, extra):
    cases = load_cases(cases_path)
    os.makedirs(out, exist_ok=True)
    tail = max(len(c['ref']) for c in cases) + 1
    env = dict(os.environ, STRATA_SERVE_DUMP=out, STRATA_SERVE_DUMP_TAIL=str(tail), STRATA_NO_LARGEPAGES='1')
    # memguard.sh backgrounds its command (`&`), which gives it /dev/null as stdin - fatal for --serve.  So launch as the
    # production launcher does (tools/v41/serve/ds41-engine.sh): the shared GPU lock + a RAM-capped scope, exec'd, with
    # memguard's MemAvailable precheck done here.
    avail = int(next(l for l in open('/proc/meminfo') if l.startswith('MemAvailable')).split()[1]) // 1048576
    if avail < 75:
        sys.exit(f'refusing: MemAvailable {avail} GiB < 75 GiB')
    lock = os.environ.get('DS4_LOCK', os.path.expanduser('~/.quetza-data/conductor/ds4-gpu.lock'))
    cmd = ['flock', lock, 'systemd-run', '--user', '--scope', '-q', f'--unit=ds41-quality-{os.getpid()}', '-p', 'MemoryMax=84G',
           '-p', 'MemorySwapMax=0', 'build-ds4-dspark/ds41_generate'] + BASE_ARGS + extra
    log = open(out + '/engine.log', 'w')
    p = subprocess.Popen(cmd, cwd=DS, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)

    def until(prefixes):
        while True:
            ln = p.stdout.readline()
            if not ln:
                raise RuntimeError('engine exited (see engine.log)')
            if ln.startswith(prefixes):
                return ln.strip()

    print(until(('READY',)), flush=True)
    t0 = time.time()
    meta = []
    for i, c in enumerate(cases):
        p.stdin.write('GEN 1 ' + ','.join(map(str, c['ids'])) + '\n')
        p.stdin.flush()
        r = until(('DONE', 'ERR'))
        meta.append({'id': c['id'], 'req': i, 'result': r})
        if r.startswith('ERR'):
            print(f'{c["id"]}: {r}', flush=True)
        if i % 10 == 0:
            print(f'{i}/{len(cases)} {c["id"]} ({len(c["ids"])} tok) {time.time() - t0:.0f}s', flush=True)
    p.stdin.write('QUIT\n'); p.stdin.flush()
    p.wait(timeout=120)
    json.dump({'tail': tail, 'extra': extra, 'env': {k: v for k, v in os.environ.items() if k.startswith('STRATA_')},
               'meta': meta, 'seconds': time.time() - t0}, open(out + '/run.json', 'w'), indent=1)
    print(f'done in {time.time() - t0:.0f}s', flush=True)


def logsoftmax(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def score(cases_path, out):
    cases = load_cases(cases_path)
    rj = json.load(open(out + '/run.json'))
    tail = rj['tail']
    groups = {}
    allr = []
    # req files are numbered by row of the cases file the run used; map by case id so a subset (short34) can score a
    # superset run (short)
    req_of = {m['id']: m['req'] for m in rj.get('meta', [])}
    for i, c in enumerate(cases):
        f = f'{out}/req{req_of.get(c["id"], i):03d}.tail.f32'
        if not os.path.exists(f):
            continue
        T = np.fromfile(f, dtype=np.float32).reshape(-1, NV)
        k = len(c['ref'])
        # the request's delta tokens are prompt + continuation; the tail holds its last min(tail, delta) rows, the last
        # row predicting past the continuation.  Row for continuation token j = T[-(k + 1) + j].
        rows = T[-(k + 1):-1]
        nll = top1 = 0.0
        ovl = 0.0
        for j in range(k):
            ls = logsoftmax(rows[j].astype(np.float64))
            nll += -ls[c['ref'][j]]
            top1 += int(np.argmax(rows[j]) == c['api_top1'][j])
            seen = set()
            for tid, lpa in c['api_top20'][j]:
                if tid < 0 or tid in seen:
                    continue
                seen.add(tid)
                ovl += min(np.exp(ls[tid]), np.exp(lpa))
        rec = (nll / k, top1 / k, ovl / k, k)
        g = 'long' if c['n_prompt'] >= 2048 else 'short'
        groups.setdefault(g, []).append(rec)
        allr.append(rec)

    def agg(rs):
        n = sum(r[3] for r in rs)
        return (sum(r[0] * r[3] for r in rs) / n, sum(r[1] * r[3] for r in rs) / n, sum(r[2] * r[3] for r in rs) / n, len(rs), n)

    res = {'all': agg(allr), **{g: agg(rs) for g, rs in groups.items()}}
    return res


def main():
    mode, cases_path = sys.argv[1], sys.argv[2]
    if mode == 'run':
        run(cases_path, sys.argv[3], sys.argv[4:])
    else:
        print(f"{'run':40s} {'group':6s} {'NLL':>8s} {'top-1':>7s} {'overlap':>8s} {'cases':>5s} {'pos':>5s}")
        for out in sys.argv[3:]:
            for g, (nll, t1, ov, nc, npos) in score(cases_path, out).items():
                print(f'{os.path.basename(out.rstrip("/")):40s} {g:6s} {nll:8.4f} {t1*100:6.2f}% {ov:8.4f} {nc:5d} {npos:5d}')


if __name__ == '__main__':
    main()
