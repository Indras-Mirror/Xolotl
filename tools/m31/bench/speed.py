#!/usr/bin/env python3
"""Portable om_bench recipe: warmup, 4K/16K/32K, greedy 256-token chat requests.

Uses the local M31 tokenizer export, or reference /tokenize and /detokenize.
Generation always goes through /v1/chat/completions on the server under test.
Prompt lengths include the chat template, request nonce and final instruction.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import sys
import urllib.request
import uuid

SIZES = {"warmup": (1024, 32), "4K": (4096, 256), "16K": (16384, 256), "32K": (32768, 256)}
ASK = "\n\n---\nПерескажи подробно по-русски, о чём этот текст: не меньше 400 слов, по разделам."


def post(args, path, body):
    request = urllib.request.Request(args.url.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    if args.api_key:
        request.add_header("Authorization", "Bearer " + args.api_key)
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        return json.load(response), time.monotonic() - started


def timing_metrics(response):
    """Normalize both servers to timed prompt work and n-1 decode transitions.

    The first generated token comes from the prompt's last hidden row. Its head
    and sampling belong to prompt_ms. predicted_ms covers subsequent transitions.
    Keep the API's reported rates too, without trusting their token denominator.
    """
    timings, usage = response.get("timings") or {}, response.get("usage") or {}
    prompt_n = timings.get("prompt_n", usage.get("prompt_tokens"))
    generated = timings.get("predicted_n", usage.get("completion_tokens"))
    prompt_ms = timings.get("prompt_ms", usage.get("prompt_ms"))
    decode_ms = timings.get("predicted_ms", usage.get("decode_ms"))
    if type(prompt_n) is not int or prompt_n <= 0 or type(generated) is not int or generated < 0:
        raise ValueError("server did not expose valid timed prompt/generated token counts")
    for name, value in (("prompt_ms", prompt_ms), ("predicted_ms", decode_ms)):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("server did not expose a valid " + name)
    transitions = max(0, generated - 1)
    if prompt_ms <= 0 or transitions and decode_ms <= 0:
        raise ValueError("server returned zero time for nonempty measured work")
    return {"prompt_n": prompt_n, "generated": generated, "prompt_ms": prompt_ms, "decode_ms": decode_ms,
            "decode_steps": transitions, "prefill_tps": 1000 * prompt_n / prompt_ms,
            "gen_tps": 1000 * transitions / decode_ms if transitions else 0.0,
            "raw_prefill_tps": timings.get("prompt_per_second"),
            "raw_gen_tps": timings.get("predicted_per_second")}


def local_tokenizer(directory):
    # Import the adapter only, without importing or patching the common server.
    root = Path(__file__).resolve().parents[3]
    sys.path[:0] = [str(root / "tools"), str(root / "tools/m31/serve")]
    from m31_tok import M31Tokenizer
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    merges = (directory / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((directory / "token_type.json").read_text())
    return M31Tokenizer(tokens, merges, types)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="minimax-m31")
    p.add_argument("--api-key", default="")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, help="M31 export directory; required for common-server runs")
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--nonce-seed", help="repeatable request nonces for matched ABBA bodies; omitted uses uuid4")
    a = p.parse_args()
    if a.repetitions <= 0 or a.timeout <= 0:
        p.error("repetitions and timeout must be positive")
    a.out.mkdir(parents=True, exist_ok=False)
    content = a.corpus.read_text(encoding="utf-8")[:1200000]
    tokenizer = local_tokenizer(a.tokenizer) if a.tokenizer else None
    tokens = tokenizer.encode(content) if tokenizer else post(a, "/tokenize", {"content": content})[0]["tokens"]
    if len(tokens) < 32768:
        p.error("corpus tokenization has fewer than 32,768 tokens")
    manifest = {"format": "m31-om-bench-v1", "recipe": "om_bench greedy low", "repetitions": a.repetitions,
                "corpus_sha256": hashlib.sha256(a.corpus.read_bytes()).hexdigest(),
                "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "tokenized_corpus_tokens": len(tokens), "complete": False,
                "tokenization": "local M31 export" if tokenizer else "reference HTTP tokenizer",
                "timing_contract": {"prefill_tps": "1000 * timed prompt_n / prompt_ms",
                    "gen_tps": "1000 * max(generated - 1, 0) / predicted_ms",
                    "first_token": "head and first sample included in prompt_ms",
                    "raw_rates_retained": True}}
    if a.nonce_seed is not None:
        manifest["nonce_seed"] = a.nonce_seed
    if tokenizer:
        manifest["tokenizer_sha256"] = {name: hashlib.sha256((a.tokenizer / name).read_bytes()).hexdigest()
            for name in ("vocab.json", "merges.txt", "token_type.json")}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (a.out / "results.tsv").open("x") as output, (a.out / "raw.jsonl").open("x") as raw:
        output.write("round\tdepth\tprompt_n\tprefill_tps\tprefill_s\tgen_n\tgen_tps\twall_s\tfinish\tdecode_steps\tdecode_s\traw_prefill_tps\traw_gen_tps\n")
        for k, label in enumerate(["warmup"] + ["4K", "16K", "32K"] * a.repetitions):
            count, maximum = SIZES[label]
            start = (k * 7919) % max(1, len(tokens) - count)
            selected = tokens[start:start + count]
            chunk = tokenizer.decode(selected) if tokenizer else post(a, "/detokenize", {"tokens": selected})[0]["content"]
            nonce = uuid.uuid4() if a.nonce_seed is None else uuid.UUID(
                bytes=hashlib.sha256(f"{a.nonce_seed}-{k}".encode()).digest()[:16])
            body = {"model": a.model, "messages": [{"role": "user", "content": f"[{nonce.hex}]\n" + chunk + ASK}],
                    "temperature": 0, "max_tokens": maximum, "ignore_eos": True, "stream": False,
                    "cache_prompt": False, "reasoning_effort": "low"}
            response, wall = post(a, "/v1/chat/completions", body)
            raw.write(json.dumps({"depth": label, "request": body, "response": response, "wall_seconds": wall}, ensure_ascii=False) + "\n")
            raw.flush()
            metrics = timing_metrics(response)
            generated = metrics["generated"]
            row = [(k - 1) // 3 + 1 if k else 0, label, metrics["prompt_n"], metrics["prefill_tps"], metrics["prompt_ms"] / 1000,
                   generated, metrics["gen_tps"], wall, response["choices"][0].get("finish_reason"), metrics["decode_steps"],
                   metrics["decode_ms"] / 1000, metrics["raw_prefill_tps"], metrics["raw_gen_tps"]]
            if generated != maximum:
                raise ValueError(f"expected {maximum} generated tokens, got {generated}; raw response retained")
            print("\t".join(map(str, row)), flush=True)
            if k:
                output.write("\t".join(map(str, row)) + "\n")
                output.flush()
    manifest["complete"] = True
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
