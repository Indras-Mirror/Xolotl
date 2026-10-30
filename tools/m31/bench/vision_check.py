#!/usr/bin/env python3
"""Deterministic media fixtures and actual HTTP/SSE vision acceptance probes.

prepare only writes owned fixtures/plans. http measures end-to-end first generated SSE delta and first visible content,
saves complete request/response bodies and checks visible facts. It starts no
server. Run each cold/warm pair in the coordinator-owned ABBA server window;
embedding_cache_mb=0 is required for a warm real encode instead of an artifact hit.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit



def image(path):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()}}


def text(value):
    return {"type": "text", "text": value}


def prepare(directory, reference_image, reference_video, font_name="DejaVuSans.ttf"):
    from PIL import Image, ImageDraw, ImageFont
    directory.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype(font_name, 72)
    canvas = Image.new("RGB", (640, 360), "white")
    ImageDraw.Draw(canvas).text((65, 135), "M31-7429", fill="black", font=font)
    canvas.save(directory / "ocr.png")
    canvas = Image.new("RGB", (640, 360), "white")
    ImageDraw.Draw(canvas).rectangle((240, 100, 400, 260), fill=(0, 180, 0))
    canvas.save(directory / "green-square.png")
    canvas = Image.new("RGB", (640, 360), "white")
    drawing = ImageDraw.Draw(canvas)
    for x in (80, 260, 440):
        drawing.ellipse((x, 120, x + 100, 220), fill="orange")
    canvas.save(directory / "count-three.png")
    ref = image(reference_image)
    cases = [
        {"name": "ref-image", "messages": [{"role": "user", "content": [ref, text("Describe the shape, its color, location, and the digit shown. Answer in English.")]}],
         "expect": ["red", "circle", r"\b1\b|\bone\b"]},
        {"name": "ref-video", "messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "file://" + str(reference_video.resolve())}}, text("Briefly describe all three stages in chronological order: colors, shapes and positions. Answer in English.")]}],
         "expect": ["red", "circle", "green", "square", "blue", "triangle"], "ordered": ["red", "green", "blue"]},
        {"name": "ocr", "messages": [{"role": "user", "content": [image(directory / "ocr.png"), text("Read the text exactly. Give only the text.")]}], "expect": ["M31-7429"]},
        {"name": "multiple-images", "messages": [{"role": "user", "content": [ref, image(directory / "green-square.png"), text("Describe the first and second images separately, in this order. Name colors and shapes. Answer in English.")]}], "expect": ["red", "circle", "green", "square"], "ordered": ["red", "green"]},
        {"name": "image-in-dialogue", "messages": [{"role": "user", "content": "I will send a picture and later ask about it."}, {"role": "assistant", "content": "Ready."}, {"role": "user", "content": [image(directory / "green-square.png"), text("Remember this picture for my next question.")]}, {"role": "assistant", "content": "I have seen the picture. What would you like to know?"}, {"role": "user", "content": "What color and shape was shown in the earlier picture? Answer in English."}], "expect": ["green", "square"]},
        {"name": "count-shapes", "messages": [{"role": "user", "content": [image(directory / "count-three.png"), text("How many orange circles are visible? Answer with the number and shape in English.")]}], "expect": [r"\b3\b|\bthree\b", "circle"]},
    ]
    low = [
        {"name": "image", "messages": [{"role": "user", "content": [
            text("Describe the visible shape, its color and its position. Be concise."), image(reference_image)]}]},
        {"name": "video", "messages": [{"role": "user", "content": [
            text("Describe the sequence of changes in this video in chronological order: the shape, color and position in each stage. Be concise."),
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(reference_video.read_bytes()).decode()}}]}]},
    ]
    (directory / "ttft-cases.json").write_text(json.dumps(low, ensure_ascii=False, indent=2) + "\n")
    (directory / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    return {"directory": str(directory), "cases": [case["name"] for case in cases]}


def check(case, content):
    missing = [pattern for pattern in case["expect"] if not re.search(pattern, content, re.I)]
    ordered = case.get("ordered", [])
    indices = [content.casefold().find(word) for word in ordered]
    order_ok = all(index >= 0 for index in indices) and indices == sorted(indices)
    return {"pass": not missing and order_ok, "missing": missing, "order_ok": order_ok}


def request(endpoint, body, timeout):
    url = urlsplit(endpoint)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("validation endpoint must be a local HTTP server")
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    started = time.perf_counter()
    first = visible_first = None
    content, reasoning, events = [], [], []
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read().decode(errors='replace')}")
        done = False
        while line := response.readline():
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                done = True
                break
            event = json.loads(payload)
            events.append(event)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            delta = (event.get("choices") or [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                if first is None:
                    first = time.perf_counter() - started
            if delta.get("content"):
                if visible_first is None:
                    visible_first = time.perf_counter() - started
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
        if not done or first is None:
            raise RuntimeError("SSE stream missing DONE or generated text")
        return {"ttft_s": first, "visible_ttft_s": visible_first, "wall_s": time.perf_counter() - started,
                "content": "".join(content), "reasoning_content": "".join(reasoning), "events": events}
    finally:
        connection.close()


def run_http(cases_path, endpoint, model, output, timeout, only):
    cases = json.loads(cases_path.read_text())
    results = []
    for case in cases:
        if only and case["name"] not in only:
            continue
        body = {"model": model, "messages": case["messages"], "temperature": 0, "top_p": 1,
                "seed": 0, "stream": True, "max_tokens": 1200,
                "reasoning_effort": "low", "video_fps": 1, "cache_prompt": False}
        result = {"case": case["name"], "request": body, "started_epoch": time.time()}
        try:
            result.update(request(endpoint, body, timeout))
            result.update(check(case, result["content"]) if "expect" in case else {"pass": True})
        except Exception as error:
            result.update({"pass": False, "error": f"{type(error).__name__}: {error}"})
        results.append(result)
        # Preserve partial results after every request, including failed requests.
        output.write_text(json.dumps({"format": "m31-vision-http-validation-v1", "pass": all(r["pass"] for r in results), "cases": results}, ensure_ascii=False, indent=2) + "\n")
    if not results:
        raise ValueError("no cases selected")
    return {"pass": all(r["pass"] for r in results), "cases": len(results), "output": str(output)}



def json_request(endpoint, body, timeout):
    from urllib.parse import urlsplit
    url = urlsplit(endpoint)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    try:
        connection.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json'})
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {result}')
        return result
    finally:
        connection.close()


def tools_smoke(endpoint, output, timeout, model="minimax-m31"):
    cases = [
        ('get_weather', {'city':'Paris'}, {'city':{'type':'string'}},
         'Call get_weather with city exactly Paris. Do not answer without the tool call.'),
        ('add', {'a':7,'b':5}, {'a':{'type':'integer'},'b':{'type':'integer'}},
         'Call add once with a=7 and b=5. Do not compute the answer yourself.'),
        ('set_config', {'config':{'enabled':True,'labels':['red','blue']}},
         {'config':{'type':'object','properties':{'enabled':{'type':'boolean'},'labels':{'type':'array','items':{'type':'string'}}},'required':['enabled','labels'],'additionalProperties':False}},
         'Call set_config exactly once with config={"enabled":true,"labels":["red","blue"]}.'),
    ]
    results = []
    for name, expected, properties, prompt in cases:
        body = {'model':model,'messages':[{'role':'user','content':prompt}], 'temperature':0, 'seed':0,
                'max_tokens':512,'tool_choice':'auto','parallel_tool_calls':True,
                'chat_template_kwargs':{'enable_thinking':False},
                'tools':[{'type':'function','function':{'name':name,'description':'Requested test function',
                          'parameters':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}}]}
        result = {'case':name,'expected':expected}
        try:
            result['response'] = json_request(endpoint,body,timeout)
            message = result['response']['choices'][0]['message']
            calls = message.get('tool_calls',[])
            if len(calls) != 1:
                result.update({'pass':False, 'error':f'expected one tool call, got {len(calls)}'})
            else:
                function = calls[0]['function']
                arguments = function['arguments']
                arguments = json.loads(arguments) if isinstance(arguments,str) else arguments
                result.update({'pass':function['name']==name and arguments==expected,'arguments':arguments})
        except Exception as exc:
            result.update({'pass':False,'error':f'{type(exc).__name__}: {exc}'})
        results.append(result)
        output.write_text(json.dumps({'pass':all(r['pass'] for r in results),'cases':results},ensure_ascii=False,indent=2)+'\n')
    return all(r['pass'] for r in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("directory", type=Path)
    prepare_parser.add_argument("--reference-image", required=True, type=Path)
    prepare_parser.add_argument("--reference-video", required=True, type=Path)
    prepare_parser.add_argument("--font", default="DejaVuSans.ttf", help="font filename or installed font name")
    http = commands.add_parser("http")
    http.add_argument("cases", type=Path)
    http.add_argument("--endpoint", default="http://127.0.0.1:8181")
    http.add_argument("--model", default="m31-vision")
    http.add_argument("--output", required=True, type=Path)
    http.add_argument("--timeout", type=float, default=600)
    http.add_argument("--only", action="append")
    tools = commands.add_parser("tools")
    tools.add_argument("--endpoint", default="http://127.0.0.1:8181")
    tools.add_argument("--model", default="minimax-m31")
    tools.add_argument("--output", required=True, type=Path)
    tools.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.mode == "tools":
        return 0 if tools_smoke(args.endpoint, args.output, args.timeout, args.model) else 1
    result = prepare(args.directory.resolve(), args.reference_image, args.reference_video, args.font) if args.mode == "prepare" else run_http(args.cases, args.endpoint, args.model, args.output, args.timeout, args.only)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
