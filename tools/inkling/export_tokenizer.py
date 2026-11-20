#!/usr/bin/env python3
"""Export Inkling-2-Mini's tokenizer into the directory layout Strata's server reads.

The layout is the one `serve/server.py` loads (docs/pack-format.md 6): `vocab.json`, `merges.txt`,
`token_type.json` (llama.cpp's: 3 = control (added, special), 4 = user-defined (added, not special), 1 = normal),
and the pack's `tokenizer.json` marker with `pre` (the pre-tokenizer the server's BPE uses) and `special_ids`.

Inkling-2-Mini is byte-level BPE (tiktoken-style, GPT-4o's o200k pattern), no BOS, eos = eot = 200006
`<|content_model_end_sampling|>`, `<|end_message|>` = 200010, and 60 added tokens from 199998 up.  The source
is the HF checkpoint's tokenizer.json / tokenizer_config.json / special_tokens_map.json; the pack's tables are
checked to be the GGUF's (`tokenizer.ggml.*`) when the GGUF is next to the checkpoint.

  export_tokenizer.py [HF_DIR] [OUT_DIR]              (needs only the standard library)
  export_tokenizer.py --verify [OUT_DIR]              (needs `tokenizers`: encode parity vs HF, 200 strings)
"""
import json
import sys
from pathlib import Path

HF = "/media/mal/NVME1TB/Xolotl/inkling-2-mini/Inkling-2-Mini"
OUT = "/media/mal/NVME1TB/Xolotl/inkling-2-mini/pack/tokenizer"
GGUF = "/media/mal/NVME1TB/Xolotl/inkling-2-mini/Inkling-2-Mini-NVFP4.gguf"

# The pre-tokenizer: the checkpoint's tokenizer.json Split regex is GPT-4o's (o200k_base) pattern, which
# tools/strata_tokenizer.py implements as `llama4` (LLAMA_VOCAB_PRE_TYPE_GPT4O).  The GGUF names it
# `inkling2`; the alias is declared there.
PRE = "inkling2"
# The exact string in the checkpoint's pre_tokenizer.pretokenizers[0].pattern.Regex (kept verbatim for the record).
PRE_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def read_pack(hf: Path) -> dict:
    tj = json.loads((hf / "tokenizer.json").read_text(encoding="utf-8"))
    if tj["model"]["type"] != "BPE":
        raise SystemExit(f"unsupported tokenizer model {tj['model']['type']!r}")
    if tj["model"].get("byte_fallback"):
        raise SystemExit("byte_fallback is not the byte-level mapping this pack uses")
    vocab, added = tj["model"]["vocab"], tj["added_tokens"]
    n = max([len(vocab)] + [a["id"] + 1 for a in added])
    tokens = [None] * n
    for tok, i in vocab.items():
        tokens[i] = tok
    for a in added:
        if tokens[a["id"]] is not None:
            raise SystemExit(f"id {a['id']} is both a BPE token and an added token")
        tokens[a["id"]] = a["content"]
    if any(t is None for t in tokens):
        gaps = [i for i, t in enumerate(tokens) if t is None][:10]
        raise SystemExit(f"the vocabulary has gaps at {gaps}")
    special = {a["id"] for a in added if a["special"]}
    added_ids = {a["id"] for a in added}
    merges = [" ".join(m) if isinstance(m, list) else m for m in tj["model"]["merges"]]
    ids = {a["content"]: a["id"] for a in added}
    try:
        eos, end_message = ids["<|content_model_end_sampling|>"], ids["<|end_message|>"]
    except KeyError as e:
        raise SystemExit(f"the checkpoint does not name {e.args[0]}") from None
    return {"tokens": tokens, "merges": merges,
            "types": [3 if i in special else 4 if i in added_ids else 1 for i in range(n)],
            "eos": eos, "end_message": end_message}


def check_gguf(pack: dict, gguf: Path) -> None:
    """The GGUF's tables must be the same ones: the engine and the server must agree on every id."""
    if not gguf.exists():
        print(f"gguf: {gguf} is not there, the tables are not cross-checked")
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gguf_reader import GGUFFile
    meta = GGUFFile(gguf).metadata
    for key, ours in (("tokenizer.ggml.tokens", pack["tokens"]),
                      ("tokenizer.ggml.merges", pack["merges"]),
                      ("tokenizer.ggml.token_type", pack["types"])):
        theirs = meta[key]
        if len(theirs) != len(ours):
            raise SystemExit(f"{key}: {len(theirs)} in the GGUF, {len(ours)} in the pack")
        bad = next((i for i, (a, b) in enumerate(zip(theirs, ours)) if a != b), None)
        if bad is not None:
            raise SystemExit(f"{key}: entry {bad} differs: GGUF {theirs[bad]!r} vs pack {ours[bad]!r}")
    eos = meta.get("tokenizer.ggml.eos_token_id")
    if eos is not None and eos != pack["eos"]:
        raise SystemExit(f"eos: the GGUF says {eos}, the checkpoint {pack['eos']}")
    print(f"gguf: tokens, merges, token_type and eos are the same ({len(pack['tokens'])} tokens)")


def export(hf: Path, out: Path) -> None:
    pack = read_pack(hf)
    tcfg = json.loads((hf / "tokenizer_config.json").read_text(encoding="utf-8"))
    smap = json.loads((hf / "special_tokens_map.json").read_text(encoding="utf-8"))
    if smap.get("eos_token") != "<|content_model_end_sampling|>":
        raise SystemExit(f"unexpected eos_token {smap.get('eos_token')!r}")
    if tcfg.get("model_max_length") is not None and tcfg.get("bos_token") is not None:
        raise SystemExit(f"Inkling has no BOS token, the checkpoint names {tcfg['bos_token']!r}")
    check_gguf(pack, Path(GGUF))
    tokens, merges = pack["tokens"], pack["merges"]
    out.mkdir(parents=True, exist_ok=True)
    marker = {
        "model": "gpt2",
        "pre": PRE,
        "vocab_size": len(tokens),
        "n_merges": len(merges),
        "special_ids": {
            "eos": pack["eos"],                                # what serve/server.py reads (set_family)
            "eot": pack["eos"],
            "end_message": pack["end_message"],
            "tokenizer.ggml.eos_token_id": pack["eos"],
            "tokenizer.ggml.eot_token_id": pack["eos"],
            "tokenizer.ggml.padding_token_id": tok_id(tokens, "<|endoftext|>"),
        },
        "add_bos_token": False,
        "pre_pattern": PRE_PATTERN,
        "pre_pattern_source": "the checkpoint's tokenizer.json pre_tokenizer (GPT-4o/o200k pattern; "
                              "the GGUF calls this pre 'inkling2')",
    }
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(pack["types"]))
    (out / "tokenizer.json").write_text(json.dumps(marker, ensure_ascii=False, indent=1))
    print(f"{len(tokens)} tokens ({sum(1 for t in pack['types'] if t == 3)} special), {len(merges)} merges "
          f"-> {out}")


def tok_id(tokens: list[str], text: str) -> int:
    return tokens.index(text)


# ---------------------------------------------------------------------------------------------- verification
def corpus() -> list[str]:
    """200 strings: ASCII, contractions, unicode, emoji, code, JSON, whitespace runs, digits, and the
    chat-template specials (encoded with parse_special, as serve/server.py encodes prompts)."""
    import random
    import string as s
    fixed = [
        "Hello, world!", "What is the capital of Japan? One word.", "The quick brown fox jumps over the lazy dog.",
        "def f(x):\n    return x + 1\n", "class A:\n    pass\n", "if (x != y) { return null; }",
        "import torch\nprint(torch.cuda.is_available())", "for i in range(10):\n\tprint(i)",
        "const x = {a: 1, b: [2, 3]};", '{"name":"get_weather","args":{"city":"Melbourne","days":3}}',
        "https://example.com/a/b?c=d&e=f#g", "user@example.com", "/media/Crucial1TB/xolotl-inkling/serve/server.py",
        "日本語のテキストです。", "中文测试，标点符号。", "한국어 테스트", "Καλημέρα κόσμε", "Привет, мир",
        "naïve café résumé", "e\u0301 combining", "Ångström ±0.5 μm", "emoji 🙂🚀🔥", "👨‍👩‍👧‍👦 family",
        "  leading and trailing  ", "\n\n\n", "\t\tTabs\t", "a" * 50, "x" * 3 + " " * 5 + "y",
        "1234567890123", "007 42 3.14159", "0x1F 0b101 1e-9", "Don't worry, it's fine.",
        "I'll say they're not going, won't we?", "¡Hola! ¿Qué tal? — «guillemets» …", "line1\r\nline2\r\n",
        "`code` **bold** _italic_ [link](url)", "<|message_user|>hi<|end_message|>",
        "<|message_model|><|content_thinking|>think<|end_message|>",
        "<|message_model|><|content_text|>answer<|end_message|>",
        "<|content_model_end_sampling|>", '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Melbourne"}}<|end_message|>',
        "<|endoftext|>", "<|begin_of_text|>", "<|content_tool_error|>", "<|content_xml|><a b='c'/>",
        "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>",
        "Möbius 〇 ㍿ ﬁ ① ½", "𝕳𝖊𝖑𝖑𝖔 𝕨𝕠𝕣𝕝𝕕", "mixed 中文 and English 123", "emoji:🏳️‍🌈",
    ]
    rnd = random.Random(7)
    words = ["the", "Hello", "world", "function", "return", "cat", "running", "configuration", "tokenizer",
             "MoE", "llama", "xolotl", "Inkling", "supercalifragilistic", "antidisestablishmentarianism",
             "JSON", "OpenAI", "2026", "0x1F", "C++", "don't", "I'll", "they're", "café", "naïve"]
    out = list(fixed)
    while len(out) < 150:
        out.append(" ".join(rnd.choice(words) for _ in range(rnd.randint(1, 8))))
    chars = s.ascii_letters + s.digits + "_-./:"
    while len(out) < 170:
        out.append("".join(rnd.choice(chars) for _ in range(rnd.randint(3, 30))))
    while len(out) < 185:
        out.append("".join(rnd.choice(s.punctuation + " \n\t") for _ in range(rnd.randint(3, 40))))
    while len(out) < 200:
        out.append("".join(rnd.choice("áéíóúüñç日本語中文한국어🙂🚀Ωμαθηματικά") for _ in range(rnd.randint(2, 15))))
    return out


def verify(out: Path) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import strata_tokenizer as ST
    from tokenizers import Tokenizer as HFTokenizer
    vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (out / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((out / "token_type.json").read_text())
    marker = json.loads((out / "tokenizer.json").read_text(encoding="utf-8"))
    ink = ST.Tokenizer(tokens, merges, types, marker["pre"], marker.get("special_ids"))
    hf = HFTokenizer.from_file(str(Path(HF) / "tokenizer.json"))
    texts = corpus()
    same = 0
    for text in texts:
        ours = ink.encode(text, parse_special=True)
        theirs = hf.encode(text, add_special_tokens=False).ids
        if ours != theirs:
            print(f"MISMATCH {text!r}\n  pack {ours[:16]}\n  hf   {theirs[:16]}")
            continue
        if ink.decode(ours) != hf.decode(theirs, skip_special_tokens=False):
            print(f"MISMATCH on decode {text!r}")
            continue
        same += 1
    print(f"tokenizer: {same}/{len(texts)} identical (encode ids, parse_special=True, vs HF tokenizers "
          f"add_special_tokens=False)")
    return 0 if same == len(texts) else 1


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--verify":
        return verify(Path(argv[1]) if len(argv) > 1 else Path(OUT))
    args = [a for a in argv if a != "--verify"]
    hf = Path(args[0]) if len(args) > 0 else Path(HF)
    out = Path(args[1]) if len(args) > 1 else Path(OUT)
    export(hf, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
