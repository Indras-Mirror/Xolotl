#!/usr/bin/env python3
"""Opt-in image-content check against a running DS41 API server (requires Pillow)."""
import argparse
import base64
import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw


def picture(shape, color):
    image = Image.new("RGB", (560, 560), "white")
    draw = ImageDraw.Draw(image)
    if shape == "square":
        draw.rectangle((90, 90, 470, 470), fill=color)
    else:
        draw.ellipse((90, 90, 470, 470), fill=color)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8150")
    parser.add_argument("--model", default="deepseek-v4.1-flash-strata-gsq")
    parser.add_argument("--api-key-file", type=Path, help="file containing the server's API key")
    args = parser.parse_args()
    headers = {"Content-Type": "application/json"}
    if args.api_key_file:
        headers["Authorization"] = "Bearer " + args.api_key_file.read_text().strip()

    def run(content):
        body = {"model": args.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": 128, "temperature": 0.8, "reasoning_effort": "none", "seed": 42}
        req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                     headers)
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error
        print(json.dumps({"seconds": round(time.monotonic() - start, 2), "response": result}), flush=True)
        return result["choices"][0]["message"].get("content") or ""

    red, blue = picture("square", "red"), picture("circle", "blue")
    prompt = {"type": "text", "text": "What color and geometric shape are shown? Answer in a short phrase."}
    for source, words in ((red, ("red", "square")), (blue, ("blue", "circle"))):
        answer = run([prompt, {"type": "image_url", "image_url": {"url": source}}]).lower()
        if not all(word in answer for word in words):
            raise RuntimeError(f"expected {words}, received {answer!r}")
    answer = run([{"type": "text", "text": "Describe the color and shape in image 1 and image 2, in that order. Be brief."},
                  {"type": "image_url", "image_url": {"url": red}},
                  {"type": "image_url", "image_url": {"url": blue}}]).lower()
    if not all(word in answer for word in ("red", "square", "blue", "circle")):
        raise RuntimeError(f"multi-image check failed: {answer!r}")
    answer = run("What is 7 plus 8? Answer with only the number.").strip()
    if "15" not in answer:
        raise RuntimeError(f"text after images failed: {answer!r}")
    print("PASS: image differences, multiple images, and text after images")


if __name__ == "__main__":
    main()
