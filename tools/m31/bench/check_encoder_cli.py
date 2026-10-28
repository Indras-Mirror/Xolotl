#!/usr/bin/env python3
"""Independent stock PR CLI encoder dump check; caller-owned exclusive GPU window only.

This script reserves no device. The caller must reserve its GPU before running.
The CLI uses its normal mtmd caller. Stop after its first complete encoder dump
so the CPU LLM image prefill need not finish. No native arithmetic is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-manifest', type=Path, required=True)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--mmproj', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    expected_manifest = args.expected_manifest
    expected = json.loads(expected_manifest.read_text())
    # The dump path is overwritten on every encoder pass; the monitor consumes
    # the first complete image output and immediately interrupts the CLI.
    chunk = next(c for c in expected['chunks'] if c['type'] == 'image')
    expected_file = expected_manifest.parent/chunk['embeddings']
    expected_rows, dim = chunk['rows'], expected['n_embd']
    if expected.get('backend') != 'cuda' or dim != 6144:
        parser.error('expected manifest must contain CUDA encoder rows')
    dump = args.output/'stock-pr-embeddings.raw'
    command = [str(args.cli), '-m', str(args.model),
               '--mmproj', str(args.mmproj), '--mmproj-device', 'CUDA0',
               '--image', str(args.image), '-p', '<__media__>', '-n', '0',
               '-ngl', '0', '-c', '4096', '-b', '2048', '-ub', '2048',
               '-t', '80', '-tb', '80', '--flash-attn', 'auto', '--no-warmup', '--jinja']
    env = os.environ.copy()
    env['MTMD_DEBUG_EMBEDDINGS'] = str(dump)
    # Receipt makes caller/backend matching explicit; do not change arithmetic
    # settings here. In particular this is not the optional F32 experiment.
    receipt = dict(scope='stock PR mtmd CLI first image encoder dump vs adapter CUDA',
                   command=command, expected_manifest=str(expected_manifest),
                   encoder_environment={k: v for k, v in env.items() if k.startswith('GGML_')},
                   model_llm_placement='CPU ngl0; CUDA0 mmproj only',
                   expected_rows=expected_rows, embedding_dimension=dim)
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    started = time.monotonic()
    complete = False
    escalation = []
    with (args.output/'stock-cli.log').open('wb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while time.monotonic()-started < args.timeout:
                if dump.is_file() and dump.stat().st_size == 8+expected_rows*dim*4:
                    with dump.open('rb') as source:
                        header = source.read(8)
                    if struct.unpack('<ii', header) != (expected_rows, dim):
                        raise RuntimeError('stock PR encoder dump shape differs')
                    complete = True
                    break
                if process.poll() is not None:
                    break
                time.sleep(.02)
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                escalation.append('SIGINT')
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate(); escalation.append('SIGTERM')
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); escalation.append('SIGKILL'); process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill(); escalation.append('SIGKILL-finally'); process.wait(timeout=10)
    receipt.update(stock_cli_returncode=process.returncode, stop_signals=escalation,
                   elapsed_seconds=time.monotonic()-started, dump_complete=complete)
    if not complete:
        (args.output/'summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
        raise RuntimeError('stock PR CLI did not produce a complete encoder dump; see stock-cli.log')
    raw = dump.read_bytes()[8:]
    oracle = expected_file.read_bytes()
    receipt.update(bit_identical=raw == oracle,
                   stock_payload_sha256=hashlib.sha256(raw).hexdigest(),
                   adapter_payload_sha256=hashlib.sha256(oracle).hexdigest(),
                   payload_bytes=len(raw))
    if raw != oracle:
        import numpy as np
        x = np.frombuffer(raw, dtype='<f4').astype('float64')
        y = np.frombuffer(oracle, dtype='<f4').astype('float64')
        receipt.update(max_abs=float(np.max(np.abs(x-y))),
                       relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(y), 1e-30)))
    (args.output/'summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)
    if not receipt['bit_identical']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
