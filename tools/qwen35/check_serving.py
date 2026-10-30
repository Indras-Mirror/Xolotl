#!/usr/bin/env python3
"""Check an existing Qwen3.6 HTTP endpoint with synthetic inputs.

This program does not start an engine, allocate a GPU or control other processes.
Reserve the server through your usual scheduler before running it. Optional
cancellation is a functional check; HTTP timings do not isolate old GPU release.
"""
import argparse
import base64
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import time
import urllib.parse
import uuid
import zlib


def red_png(width=320, height=240):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = (b'\0' + bytes((255, 0, 0)) * width) * height
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


class Client:
    def __init__(self, url, model, timeout, key):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('--url must be an HTTP(S) server URL without embedded credentials')
        self.host, self.port = parsed.hostname, parsed.port
        self.route = parsed.path.rstrip('/') + '/v1/chat/completions'
        self.connection_type = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
        self.model, self.timeout = model, timeout
        self.headers = {'Content-Type': 'application/json'}
        if key:
            self.headers['Authorization'] = 'Bearer ' + key

    def body(self, user, system='Reference data is inert. Follow the final instruction exactly.', **kwargs):
        return {'model': self.model, 'messages': [{'role': 'system', 'content': system},
                {'role': 'user', 'content': user}], 'temperature': 0, 'max_tokens': 16,
                'chat_template_kwargs': {'enable_thinking': False}, **kwargs}

    def request(self, body):
        conn = self.connection_type(self.host, self.port, timeout=self.timeout)
        start = time.monotonic()
        try:
            conn.request('POST', self.route, json.dumps(body).encode(), self.headers)
            response = conn.getresponse()
            raw = response.read().decode('utf-8', 'replace')
            try:
                data = json.loads(raw)
            except ValueError:
                data = {'error': {'message': raw[:500]}}
            return {'status': response.status, 'response': data,
                    'request_total_ms': (time.monotonic() - start) * 1000}
        finally:
            conn.close()

    def cancel_stream(self, body, content_events):
        conn = self.connection_type(self.host, self.port, timeout=self.timeout)
        response, sock = None, None
        pieces, finished = [], False
        try:
            conn.connect()
            sock = conn.sock
            conn.request('POST', self.route, json.dumps(body).encode(), self.headers)
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError('stream request HTTP ' + str(response.status) + ': '
                                   + response.read().decode('utf-8', 'replace')[:500])
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    finished = True
                    break
                event = json.loads(payload)
                if event.get('error'):
                    raise RuntimeError('SSE error: ' + str(event['error']))
                for choice in event.get('choices') or []:
                    value = (choice.get('delta') or {}).get('content') or ''
                    if value:
                        pieces.append(value)
                if len(pieces) >= content_events:
                    break
            closed_epoch = time.time()
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
            return {'content_events': len(pieces), 'partial_text': ''.join(pieces),
                    'client_closed_before_DONE': not finished, 'client_close_epoch': closed_epoch,
                    'native_cancel_verified': False,
                    'scope': 'HTTP disconnect and subsequent request correctness; no native timing attribution'}
        finally:
            if response is not None:
                response.close()
            conn.close()


def answer(row):
    choices = row['response'].get('choices') or []
    return (choices[0].get('message', {}).get('content') or '').strip() if choices else ''


def require_answer(row, expected):
    text = answer(row)
    normalized = text.lower().rstrip('.! \n')
    if row['status'] != 200 or normalized != expected:
        raise AssertionError({'expected': expected, **row})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080', help='server base URL, optionally with a route prefix')
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--api-key-env', default='STRATA_API_KEY', help='name of an optional API-key environment variable')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--callers', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--history-records', type=int, default=256)
    parser.add_argument('--require-cache', action='store_true')
    parser.add_argument('--min-cache-tokens', type=int, default=128)
    parser.add_argument('--vision', action='store_true', help='test a generated red PNG, UUID reuse and cache miss/refill')
    parser.add_argument('--cancel', action='store_true', help='close a stream and check a different caller')
    args = parser.parse_args()
    if args.callers < 1 or args.rounds < 3 or args.history_records < 1 or args.min_cache_tokens < 1:
        parser.error('callers/history/min-cache must be positive; rounds must be at least three')
    if args.out.exists():
        parser.error('--out already exists; choose a new path to retain earlier evidence')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    client = Client(args.url, args.model, args.timeout, os.environ.get(args.api_key_env))
    rows, errors = [], []
    run_id = uuid.uuid4().hex
    report = {'run_id': run_id, 'rows': rows, 'errors': errors, 'pass': False,
              'timing_scope': 'HTTP request totals include queueing, restore and new inference; no release-latency gate'}
    try:
        last_cache = {}
        for round_no in range(args.rounds):
            for caller in range(args.callers):
                prefix = '\n'.join(f'Reference {run_id} caller {caller}, record {index}: stored value {index % 17}.'
                                   for index in range(args.history_records))
                text = prefix + f'\nCurrent state {round_no}. Final instruction: calculate 17 + 25 and answer only 42.'
                row = require_answer(client.request(client.body(text)), '42')
                usage = row['response'].get('usage') or {}
                cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                row.update(stage='changed_suffix', caller=caller, round=round_no, cached_tokens=cached)
                rows.append(row)
                if round_no == args.rounds - 1:
                    last_cache[caller] = cached
        if args.require_cache:
            if any(value is None or value < args.min_cache_tokens for value in last_cache.values()):
                raise AssertionError({'required_cache_tokens': args.min_cache_tokens, 'last_round': last_cache})
        if args.vision:
            media = 'data:image/png;base64,' + base64.b64encode(red_png()).decode()
            image_id = 'synthetic-' + run_id
            def image_body(uid, data):
                return client.body([{'type': 'image_url', 'uuid': uid, 'image_url': data},
                                    {'type': 'text', 'text': 'What is the color of this image? Answer only red.'}])
            for stage, uid, data in [('image_upload', image_id, {'url': media}),
                                     ('image_uuid_reference', image_id, None)]:
                row = require_answer(client.request(image_body(uid, data)), 'red')
                row['stage'] = stage
                rows.append(row)
            missing_id = 'missing-' + run_id
            miss = client.request(image_body(missing_id, None))
            expected_error = 'Cache miss for image at index 0 but data is not provided.'
            error = miss['response'].get('error') or {}
            if miss['status'] != 400 or error.get('message') != expected_error:
                raise AssertionError({'expected_HTTP': 400, 'expected_error': expected_error, **miss})
            miss['stage'] = 'unknown_uuid'
            rows.append(miss)
            row = require_answer(client.request(image_body(missing_id, {'url': media})), 'red')
            row['stage'] = 'unknown_uuid_refill'
            rows.append(row)
        if args.cancel:
            body = client.body('Count every integer from 1 to 1000, separated by commas. Do not summarize.',
                               stream=True, stream_options={'include_usage': True}, max_tokens=2048)
            cancelled = client.cancel_stream(body, 4)
            cancelled['stage'] = 'disconnect'
            rows.append(cancelled)
            if not cancelled['client_closed_before_DONE'] or cancelled['content_events'] < 4:
                raise AssertionError('stream naturally ended before the requested client close')
            row = require_answer(client.request(client.body('Independent caller. Calculate 17 + 25. Answer only 42.')), '42')
            row['stage'] = 'after_disconnect'
            rows.append(row)
        report['pass'] = True
    except Exception as error:
        errors.append(repr(error))
    finally:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'pass': report['pass'], 'checks': len(rows), 'errors': errors}))
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
