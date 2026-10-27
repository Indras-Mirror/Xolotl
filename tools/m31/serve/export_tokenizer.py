#!/usr/bin/env python3
"""Export an M31 GGUF header for the common server, without loading weights.

    python tools/m31/serve/export_tokenizer.py MODEL.gguf TOKENIZER_DIR
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile  # noqa: E402


def export(model, directory):
    metadata = GGUFFile(Path(model)).metadata
    if metadata.get("tokenizer.ggml.model") != "gpt2" or metadata.get("tokenizer.ggml.pre") != "minimax-m2":
        raise ValueError("expected the M31 gpt2/minimax-m2 tokenizer")
    if metadata.get("tokenizer.ggml.normalizer.nfc") is not True:
        raise ValueError("this M31 adapter requires tokenizer.ggml.normalizer.nfc=true")
    tokens = metadata["tokenizer.ggml.tokens"]
    types = metadata["tokenizer.ggml.token_type"]
    merges = metadata["tokenizer.ggml.merges"]
    template = metadata["tokenizer.chat_template"]
    if len(types) != len(tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("inconsistent token types or duplicate vocabulary entries")
    if not isinstance(template, str) or "]<]minimax[>[" not in template:
        raise ValueError("missing MiniMax chat template")
    special_ids = {key: value for key, value in metadata.items()
                   if key.startswith("tokenizer.ggml.") and key.endswith("_token_id")}
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    files = {"vocab.json": {token: i for i, token in enumerate(tokens)},
             "token_type.json": types,
             "tokenizer.json": {"model": "gpt2", "pre": "minimax-m2", "normalizer": "NFC",
                                "vocab_size": len(tokens), "special_ids": special_ids}}
    for name, value in files.items():
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "chat_template.jinja").write_text(template, encoding="utf-8")
    return {"vocab_size": len(tokens), "merges": len(merges), "directory": str(out)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(export(args.model, args.directory)))


if __name__ == "__main__":
    main()
