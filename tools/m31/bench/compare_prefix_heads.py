#!/usr/bin/env python3
"""Strict same-history cache-head comparison; no model/backend initialization."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys

VOCAB = 200064
HEADS = ('whole', 'whole_rewind', 'whole_again', 'whole_decode_rewind', 'split',
         'split_rewind', 'split_decode_rewind', 'restore', 'release_rewind')


def expected_recipes(length, reuse=1024, decode_steps=7):
    suffix = [{'position': reuse, 'tokens': length-reuse}]
    decoded = [{'position': length+i, 'tokens': 1} for i in range(decode_steps)]
    return {'whole': [{'position': 0, 'tokens': length}], 'whole_again': [{'position': 0, 'tokens': length}],
            'whole_rewind': suffix, 'whole_decode_rewind': decoded+suffix,
            'split': [{'position': 0, 'tokens': reuse}]+suffix, 'split_rewind': suffix,
            'split_decode_rewind': decoded+suffix, 'restore': suffix, 'release_rewind': suffix}


def load_run(directory, length, *, vocab=VOCAB):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text())
    for key, value in (('length', length), ('reused', 1024), ('vocab', vocab)):
        if type(summary.get(key)) is not int or summary[key] != value:
            raise ValueError(f'{directory}: wrong {key}')
    if summary.get('same_history_bitwise') is not True:
        raise ValueError(f'{directory}: checker same-history invariants failed')
    if set(summary.get('heads', {})) != set(HEADS):
        raise ValueError(f'{directory}: missing or extra summary head')
    if {p.name for p in directory.glob('*.f32')} != {name+'.f32' for name in HEADS}:
        raise ValueError(f'{directory}: missing or extra F32 head file')
    recipes = expected_recipes(length)
    payloads, records = {}, {}
    for name in HEADS:
        declared = summary['heads'][name]
        recipe = declared.get('forward_chunks')
        if recipe != recipes[name] or any(type(x.get('position')) is not int or type(x.get('tokens')) is not int for x in recipe):
            raise ValueError(f'{directory}/{name}: changed actual chunk recipe')
        raw = (directory/(name+'.f32')).read_bytes()
        if len(raw) != vocab*4:
            raise ValueError(f'{directory}/{name}: not one full-vocabulary F32 head')
        values = array('f'); values.frombytes(raw)
        if values.itemsize != 4:
            raise RuntimeError('requires 32-bit float array storage')
        if sys.byteorder != 'little':
            values.byteswap()
        if any(not math.isfinite(v) for v in values):
            raise ValueError(f'{directory}/{name}: non-finite logits')
        top = max(range(vocab), key=values.__getitem__)
        if type(declared.get('top1')) is not int or declared['top1'] != top:
            raise ValueError(f'{directory}/{name}: top1 metadata differs from raw head')
        payloads[name] = raw
        records[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
                         'top1': top, 'forward_chunks': recipe}
    groups = (('whole', 'whole_again'), ('whole_rewind', 'whole_decode_rewind'),
              ('split', 'split_rewind', 'split_decode_rewind', 'restore', 'release_rewind'))
    for names in groups:
        if any(payloads[name] != payloads[names[0]] for name in names[1:]):
            raise ValueError(f'{directory}: raw within-checker invariant failed {names}')
    for flag, a, b in (('whole_vs_split_bitwise', 'whole', 'split'),
                        ('whole_vs_replay_bitwise', 'whole', 'whole_rewind')):
        if type(summary.get(flag)) is not bool or summary[flag] != (payloads[a] == payloads[b]):
            raise ValueError(f'{directory}: incorrect diagnostic {flag}')
    return summary, payloads, records


def compare(candidate, base, length, *, vocab=VOCAB):
    a, av, ar = load_run(candidate, length, vocab=vocab)
    b, bv, br = load_run(base, length, vocab=vocab)
    heads = {}
    for name in HEADS:
        same = av[name] == bv[name]
        changed = [i for i in range(vocab) if av[name][i*4:i*4+4] != bv[name][i*4:i*4+4]] if not same else []
        heads[name] = {'pass': same, 'bitwise_identical': same, 'candidate': ar[name], 'base': br[name],
                       'changed_vocabulary_entries': len(changed), 'first_changed_indices': changed[:16]}
        if ar[name]['forward_chunks'] != br[name]['forward_chunks']:
            raise ValueError('cross-arm actual chunk recipe differs: '+name)
    flags_equal = all(a[key] == b[key] for key in ('whole_vs_split_bitwise', 'whole_vs_replay_bitwise'))
    return {'format': 'm31-cache-numeric-comparison-v1', 'pass': all(row['pass'] for row in heads.values()) and flags_equal,
            'length': length, 'reuse': 1024, 'suffix': length-1024, 'decode_steps': 7, 'vocab': vocab,
            'heads': heads, 'within_checker_invariants': True, 'exact_actual_recipes': True,
            'diagnostic_flags_equal': flags_equal,
            'whole_vs_split_bitwise': {'A': a['whole_vs_split_bitwise'], 'B': b['whole_vs_split_bitwise']},
            'whole_vs_replay_bitwise': {'A': a['whole_vs_replay_bitwise'], 'B': b['whole_vs_replay_bitwise']},
            'scope': 'all9 complete F32 heads; strict same-history cross-arm bytes; no cross-history equality requirement'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate', type=Path); p.add_argument('base', type=Path)
    p.add_argument('--length', required=True, type=int, choices=(1124, 1324))
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    try:
        result = compare(a.candidate, a.base, a.length)
    except Exception as error:
        result = {'format': 'm31-cache-numeric-comparison-v1', 'pass': False,
                  'length': a.length, 'error': type(error).__name__+': '+str(error)}
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'pass': result['pass'], 'length': a.length, 'out': str(a.out)}))
    return 0 if result['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
