#!/usr/bin/env python3
"""Exercise real q36 through serve, including duplicate constraint fields and SSE usage."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from q36_grammar_bench import DIARY, valid, write_example_grammars
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate
from serve.server import Service, StrataEngine, Vision, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "model", "tokenizer", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--grammars", type=Path, help="optional custom scene/turn/turn_quiet GBNF directory")
    parser.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--vision-config", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.grammars is None:
        args.grammars = write_example_grammars(args.out / "grammars")
    vocab = json.loads((args.tokenizer / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = Tokenizer(tokens, (args.tokenizer / "merges.txt").read_text().split("\n"),
                    json.loads((args.tokenizer / "token_type.json").read_text()))
    report = []
    for mtp in (3, 0):
        engine = StrataEngine(str(args.engine), ["--native", str(args.model), "--max-context", "8192",
                              "--mtp", str(mtp), "--vision"], log=str(args.out / f"http-engine-{mtp}.log"),
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": args.device})
        vision = None
        httpd = None
        try:
            if args.vision_config:
                cfg = json.loads(args.vision_config.read_text())["vision"]
                cfg.pop("cuda_device", None)
                vision = Vision(cfg, log=(args.out / f"http-vision-{mtp}.log").open("a"), env={**os.environ, "CUDA_VISIBLE_DEVICES": args.device})
            svc = Service(engine, tok, ChatTemplate(args.tokenizer / "chat_template.jinja"), model_name="q36-check", vision=vision)
            svc.structured_compact = True
            httpd = serve(svc, port=0)
            url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions"

            def request(name, fields, user, kind=None, image=False, cached_image=False):
                content = user
                if image:
                    from PIL import Image, ImageDraw
                    pic = Image.new("RGB", (320, 240), "white")
                    ImageDraw.Draw(pic).rectangle((50, 50, 160, 160), fill="red")
                    data = io.BytesIO()
                    pic.save(data, format="JPEG")
                    media = "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()
                    content = [{"type": "image_url", "uuid": "q36gram-http-frame", "image_url": {"url": media}},
                               {"type": "text", "text": user}]
                    if cached_image:
                        content[0] = {"type": "image_url", "uuid": "q36gram-http-frame", "image_url": None}
                body = {"model": "q36-check", "messages": [{"role": "system", "content": "Return a brief answer in the requested format."},
                                                         {"role": "user", "content": content}],
                        "temperature": .4 if kind == "scene" else .9, "top_p": .9,
                        "max_tokens": 700 if kind in ("turn", "turn_quiet") else 200,
                        "stream": True, "stream_options": {"include_usage": True},
                        "chat_template_kwargs": {"enable_thinking": False}, **fields}
                req = urllib.request.Request(url, json.dumps(body, ensure_ascii=False).encode(),
                                             {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as response:
                    raw = response.read().decode()
                (args.out / f"http-{name}-{mtp}.sse").write_text(raw)
                chunks = [json.loads(line[6:]) for line in raw.splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                assert raw.endswith("data: [DONE]\n\n"), raw
                assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0, chunks[-1]
                assert chunks[-2]["choices"][0]["finish_reason"] == "stop", chunks[-2]
                assert "usage" not in chunks[-2], chunks[-2]
                text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c.get("choices"))
                row = {"name": name, "mtp": mtp, "text": text, "chunks": chunks, "request": body, "finish": "stop"}
                if kind:
                    assert valid(kind, row), row
                report.append(row)
                (args.out / "http-evidence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(name, "mtp=", mtp, "usage-last=OK", repr(text), flush=True)

            for kind, user in (("scene", "The room is unchanged. There is nothing to announce. Return speak=false."),
                               ("turn", "A visitor says hello. Return act=gesture;gesture=wave;say=Hello!;tone=happy"),
                               ("turn_quiet", "There is no request to handle. Return quiet.")):
                grammar = (args.grammars / (kind + ".gbnf")).read_text()
                fields = {"grammar": grammar, "structured_outputs": {"grammar": grammar, "disable_any_whitespace": True}}
                request(kind, fields, user, kind, image=kind == "scene" and vision is not None)
                if kind == "scene" and vision is not None:
                    request("scene-cached-image", fields, user, kind, image=True, cached_image=True)
            request("unconstrained-usage", {}, "Say hello in one word.")
            for name, fields in (("schema", {"response_format": {"type": "json_schema", "json_schema": {"schema": DIARY}}}),
                                 ("vllm-schema", {"structured_outputs": {"json": DIARY, "disable_any_whitespace": True}}),
                                 ("legacy-schema", {"guided_json": DIARY}), ("llama-schema", {"json_schema": DIARY})):
                request(name, fields, "Return JSON: event=Greeting, mood=happy, importance=1, people=[visitor], actions=[wave].", "diary")
                text = report[-1]["text"]
                assert json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":")) == text, text
            request("choice", {"structured_outputs": {"choice": ["yes", "no"]}}, "Is two plus two four? Answer yes or no.", "yesno")
        finally:
            if httpd:
                httpd.shutdown()
                httpd.server_close()
            engine.close()
            if vision:
                vision.close()  # retain the evidence files
    print(f"HTTP checks passed: {len(report)} requests", flush=True)


if __name__ == "__main__":
    main()
