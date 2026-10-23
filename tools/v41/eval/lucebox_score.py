#!/usr/bin/env python3
"""tools/v41/eval/lucebox_score.py - score ds41_generate --dump-logits output on Lucebox's public 1017 eval set.

Lucebox (huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF, eval/) publishes 8 x 1024 held-out token ids and
the FP4/FP8 teacher's top-256 log-probs per position.  Unsloth, dsx and Lucebox report KL / top-1 / ppl on it.  The
metric matches dsx's tools/lucebox-eval: per position KL(teacher || model) = sum_j exp(lt_j) * (lt_j - lq(idx_j)) over
the teacher's top 256 (lq = the model's full log-softmax), top-1 = model argmax == teacher's top index, ppl from the
model's NLL of the real next token.  Teacher reference: ppl 8.561 over 8184 positions.

    lucebox_score.py EVALDIR SEQ LOGITS.f32 [--acc ACC.json]   # adds one sequence (1023 scored rows) to ACC.json
    lucebox_score.py --report ACC.json
"""
import json, sys
import numpy as np

def score(evaldir, s, path, acc):
    ids = np.load(f'{evaldir}/heldout_8x1024.npy')
    z = np.load(f'{evaldir}/teacher_8x1024.npz')
    L = ids.shape[1]
    lg = np.fromfile(path, dtype=np.float32)
    nv = lg.size // L
    assert lg.size == L * nv, f'{path}: {lg.size} floats is not {L} rows'
    lg = lg.reshape(L, nv)
    r0 = s * (L - 1)
    nll = kl = 0.0; agree = 0
    for c0 in range(0, L - 1, 128):
        c1 = min(L - 1, c0 + 128)
        x = lg[c0:c1].astype(np.float64)
        lse = np.log(np.exp(x - x.max(1, keepdims=True)).sum(1)) + x.max(1)
        lq = x - lse[:, None]
        tidx = z['idx'][r0 + c0:r0 + c1]; tlp = z['lp'][r0 + c0:r0 + c1].astype(np.float64)
        kl += float((np.exp(tlp) * (tlp - np.take_along_axis(lq, tidx, 1))).sum())
        agree += int((x.argmax(1) == tidx[:, 0]).sum())
        nxt = ids[s, c0 + 1:c1 + 1]
        nll -= float(lq[np.arange(c1 - c0), nxt].sum())
    acc.setdefault('seqs', {})[str(s)] = {'tokens': L - 1, 'nll': nll, 'kl': kl, 'agree': agree}
    return acc

def report(acc):
    t = sum(v['tokens'] for v in acc['seqs'].values())
    nll = sum(v['nll'] for v in acc['seqs'].values()) / t
    out = {'seqs': len(acc['seqs']), 'tokens': t, 'nll': nll, 'ppl': float(np.exp(nll)),
           'kl_top256': sum(v['kl'] for v in acc['seqs'].values()) / t,
           'top1_agree': sum(v['agree'] for v in acc['seqs'].values()) / t}
    print(json.dumps(out))
    return out

if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == '--report':
        report(json.load(open(a[1])))
        sys.exit(0)
    evaldir, s, path = a[0], int(a[1]), a[2]
    accp = a[a.index('--acc') + 1] if '--acc' in a else None
    acc = json.load(open(accp)) if accp and __import__('os').path.exists(accp) else {}
    acc = score(evaldir, s, path, acc)
    if accp:
        json.dump(acc, open(accp, 'w'))
    report(acc)
