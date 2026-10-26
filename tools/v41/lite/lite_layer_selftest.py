#!/usr/bin/env python3
"""CPU integration gate: normal and deferred v41_calib layer-0 forwards must write identical residual/pre_mix states.
Requires the verified checkpoint and V41_REF used by lite_calib; scratch files are retained.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import numpy as np
import torch
import lite_calib as LC


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokens', required=True, help='prepared tokens.npy; the first 32 tokens of the first sequence are used')
    ap.add_argument('--out-dir', required=True, help='new scratch directory (retained)')
    a = ap.parse_args()
    torch.set_num_threads(4)
    root = Path(a.out_dir)
    root.mkdir(parents=True, exist_ok=False)
    tokens = np.load(a.tokens)[:1, :32].copy()
    assert tokens.shape == (1, 32)
    C = LC.setup_calib()
    for deferred in (False, True):
        work = root / ('deferred' if deferred else 'normal')
        work.mkdir()
        C.WORK = str(work)
        np.save(work / 'tokens.npy', tokens)
        (work / 'meta.json').write_text(json.dumps({'n': 1, 'seq': 32, 'domains': ['integration'], 'is_eval': [True]}))
        C.cmd_init(SimpleNamespace())
        pending = []
        ns = SimpleNamespace(layer=0, batch=1, device='cpu', no_gptq=True, gptq_experts=0, expert_stride=1,
                             min_fit=32, damps=[1.], screen_only=None, gc=False, defer_teacher=deferred)
        if deferred:
            def hook(L, ex, expert, xe, ge, fit, lim, out):
                # The real worker's equality is a separate gate in lite_selftest.py. Here exercise cmd_layer's
                # deferred-output plumbing, including expert order and the shared/HC update.
                pending.append((ns.cur_tok, expert(xe, ge)))
                return None, {}

            def done(L, block, stats):
                for tok, out in pending:
                    ns.moe_Y.index_add_(0, tok, out.float())
            ns.expert_hook, ns.moe_done_hook = hook, done
        C.cmd_layer(ns)
    for relative in ('h/L01.bin', 'pre/L01.npy'):
        assert (root / 'normal' / relative).read_bytes() == (root / 'deferred' / relative).read_bytes(), relative
    print('PASS: normal and deferred layer-0 residual and pre_mix states are byte-identical')


if __name__ == '__main__':
    main()
