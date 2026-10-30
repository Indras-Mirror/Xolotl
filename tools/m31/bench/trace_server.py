#!/usr/bin/env python3
"""Run the common M31 shim while recording only protocol control lines.

The caller owns GPU reservation. Both compared engines should use this same
optional wrapper. The original ready/pump methods receive unchanged lines;
T records are never written. Run speed.py with --repetitions 1, then inspect
PP chunks with trace_chunks.py.

    python tools/m31/bench/trace_server.py --trace protocol.jsonl -- \
        --engine strata --config m31.json
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

CONTROL = frozenset(('PP', 'RESUME', 'DONE', 'ERR', 'INFO', 'READY'))


class ControlTrace:
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()
        self.failed = False

    def record(self, pid, line):
        if self.failed or line.partition(' ')[0].rstrip('\r\n') not in CONTROL:
            return
        row = {'timestamp': time.time(), 'monotonic': time.monotonic(),
               'pid': pid, 'line': line.rstrip('\r\n')}
        try:
            with self.lock:
                self.stream.write(json.dumps(row) + '\n')
                self.stream.flush()
        except (OSError, ValueError) as error:
            # A failed diagnostic must not strand the real engine's stdout.
            # The driver rejects missing/incomplete traces independently.
            self.failed = True
            print('PROTOCOL_TRACE_ERROR ' + repr(error), file=sys.stderr, flush=True)


class TeeReader:
    def __init__(self, original, trace, pid):
        self.original, self.trace, self.pid = original, trace, pid

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.original)
        self.trace.record(self.pid, line)
        return line

    def readline(self, *args, **kwargs):
        line = self.original.readline(*args, **kwargs)
        if line:
            self.trace.record(self.pid, line)
        return line

    def __getattr__(self, name):
        return getattr(self.original, name)


def install_trace(module, stream):
    trace = ControlTrace(stream)
    original_ready = module.M31Engine._ready_pump
    original_pump = module.M31Engine._pump

    def ensure(proc):
        if not isinstance(proc.stdout, TeeReader):
            proc.stdout = TeeReader(proc.stdout, trace, proc.pid)

    def ready_pump(proc, out):
        ensure(proc)
        # Keep the same tee installed for _pump. Restoring it at READY would
        # race the main thread starting its next reader and could double-log.
        return original_ready(proc, out)

    def pump(self):
        ensure(self.proc)
        return original_pump(self)

    module.M31Engine._ready_pump = staticmethod(ready_pump)
    module.M31Engine._pump = pump
    return trace


def load_shim(source):
    source = Path(source).resolve(strict=True)
    sys.path[:0] = [str(source), str(source / 'tools'), str(source / 'tools/m31/serve')]
    path = source / 'tools/m31/serve/serve_m31.py'
    spec = importlib.util.spec_from_file_location('m31_traced_actual_shim', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[3],
                        help='Xolotl checkout to run; defaults to this checkout')
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('shim_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = load_shim(args.source)
    forwarded = args.shim_args[1:] if args.shim_args[:1] == ['--'] else args.shim_args
    if not forwarded:
        parser.error('supply the actual shim arguments after --')
    with args.trace.open('x', buffering=1) as stream:
        tracer = install_trace(module, stream)
        sys.argv = [str(args.source / 'tools/m31/serve/serve_m31.py'), *forwarded]
        result = module.main()
        if tracer.failed:
            raise RuntimeError('protocol trace output failed')
        return result


if __name__ == '__main__':
    raise SystemExit(main())
