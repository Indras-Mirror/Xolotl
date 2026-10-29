"""Check actual PP chunks for speed.py --repetitions 1 control traces.

Pairs a trace_server.py control trace with speed.py raw.jsonl. Requires the
uncached warmup/4K/16K/32K recipe, complete PP totals, exact output budgets and
one engine PID. This parser performs no inference and starts no process.
"""
from __future__ import annotations
import json
from pathlib import Path


def parse_groups(path, raw_path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    raw = [json.loads(line) for line in Path(raw_path).read_text().splitlines() if line.strip()]
    pids = {record.get('pid') for record in records}
    if len(pids) != 1 or any(type(pid) is not int or pid <= 0 for pid in pids):
        raise ValueError('engine restarted or invalid PID in one benchmark arm: ' + repr(pids))
    if [row['depth'] for row in raw] != ['warmup', '4K', '16K', '32K']:
        raise ValueError('expected exactly warmup/4K/16K/32K client requests')
    groups, pp, resumes = [], [], []
    for record in records:
        fields = record['line'].split()
        if not fields:
            raise ValueError('empty traced record')
        if fields[0] in ('INFO', 'READY'):
            continue
        if fields[0] == 'ERR':
            raise ValueError('engine ERR in protocol trace: ' + record['line'])
        if fields[0] == 'RESUME':
            resumes.append(int(fields[1]))
        elif fields[0] == 'PP':
            if len(fields) < 3:
                raise ValueError('short PP record')
            pp.append({'read': int(fields[1]), 'total': int(fields[2]),
                       'elapsed_ms': float(fields[3]) if len(fields) > 3 else None,
                       'reported_tps': float(fields[4]) if len(fields) > 4 else None})
        elif fields[0] == 'DONE':
            if len(groups) >= len(raw) or len(fields) < 15:
                raise ValueError('extra/short DONE record')
            generated, prompt, reused, read = int(fields[1]), int(fields[2]), int(fields[8]), int(fields[14])
            if resumes or reused != 0 or read != prompt:
                raise ValueError('cache_prompt=false benchmark reused or omitted prompt rows')
            previous, chunks = 0, []
            for row in pp:
                if row['total'] != prompt or not previous < row['read'] <= prompt:
                    raise ValueError('invalid PP row continuity/total')
                chunks.append(row['read'] - previous)
                previous = row['read']
            if not pp or previous != prompt or sum(chunks) != read:
                raise ValueError('PP chunks do not cover DONE prompt/read totals')
            client = raw[len(groups)]
            usage = client['response'].get('usage') or {}
            timings = client['response'].get('timings') or {}
            timed_prompt = timings.get('prompt_n', usage.get('prompt_tokens'))
            expected_generated = 32 if client['depth'] == 'warmup' else 256
            if (prompt != timed_prompt or generated != expected_generated or fields[5] != 'length'
                    or client['request'].get('cache_prompt') is not False):
                raise ValueError('DONE differs from exact client request/response budget')
            groups.append({'depth': client['depth'], 'prompt_n': prompt, 'read': read, 'reused': reused,
                           'actual_chunks': chunks, 'generated': generated, 'finish': fields[5],
                           'pp': pp, 'done': record['line'], 'engine_pid': record['pid']})
            pp, resumes = [], []
        else:
            raise ValueError('trace contains a non-control record: ' + record['line'])
    if len(groups) != 4 or pp or resumes:
        raise ValueError('incomplete protocol trace groups')
    return groups


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="trace_server.py JSONL output")
    parser.add_argument("--requests", type=Path, required=True, help="speed.py raw.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="new JSON result file")
    args = parser.parse_args()
    groups = parse_groups(args.trace, args.requests)
    result = {"format": "m31-protocol-chunks-v1", "pass": True,
              "scope": "uncached warmup/4K/16K/32K, one speed.py repetition; original protocol control records",
              "groups": groups}
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pass": True, "requests": len(groups),
                      "actual_chunks": {group["depth"]: group["actual_chunks"] for group in groups}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
