#!/usr/bin/env python3
"""MiniMax-M3.1 adapter for the common OpenAI/Anthropic server.

    python tools/m31/serve/serve_m31.py --engine strata --config strata-m31.json

The common server owns HTTP, streaming, queueing, cancellation, restart and
structured-output validation. This process registers only MiniMax's tokenizer,
GGUF template, output parser and native request metadata.
"""
from __future__ import annotations

import base64
import copy
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import threading
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(HERE)]

import strata_tokenizer as ST  # noqa: E402
from m31_tok import M31Tokenizer  # noqa: E402
import m31_parser as P  # noqa: E402
from serve import frontend as F  # noqa: E402
from serve import server as S  # noqa: E402
from vision_http import MediaInput, install_video_reader  # noqa: E402


_BaseVision = S.Vision


class M31Vision:
    """Deferred mtmd helper: a broken optional encoder cannot prevent text service."""
    lazy_vision, elastic = True, False

    def __init__(self, cfg, log=None, env=None, lazy=False):
        from vision_encoder import VisionEncoder
        self.dir = _BaseVision.work_dir()
        self.last_used = time.time()
        self.on_start = None
        self.options = dict(cfg)
        cache = cfg.get("cache_dir") or self.dir / "cache"
        backend = cfg.get("backend", "cuda" if cfg.get("gpu") else "cpu")
        self.client = VisionEncoder(cfg.get("exe", ""), cfg.get("model", ""), cfg.get("mmproj", ""), cache,
                                    backend=backend, threads=cfg.get("threads", 16),
                                    timeout=cfg.get("timeout", 180), startup_timeout=cfg.get("startup_timeout", 180),
                                    embedding_cache_mb=cfg.get("embedding_cache_mb", 256), env=env)
        limits = {key: value for key, value in cfg.items() if key in (
            "max_media_bytes", "max_total_bytes", "max_media", "max_image_pixels", "max_video_seconds",
            "max_video_frames", "max_video_fps", "video_fps", "allow_http", "allow_local")}
        self.media = MediaInput(cache, **limits)
        self.log = log

    def alive(self):
        proc = getattr(self.client, "process", None)
        return proc is not None and proc.poll() is None

    def encode_prompt(self, prompt, media, fps, cancel=None):
        self.last_used = time.time()
        if not self.alive() and self.on_start:
            self.on_start()
        try:
            return self.client.encode(prompt, media, video_fps=fps,
                                      timestamp_interval_ms=self.options.get("timestamp_interval_ms", 5000),
                                      cancel=cancel)
        except (OSError, RuntimeError, ValueError) as exc:
            if cancel is not None and cancel.is_set():
                raise S.ClientGone() from None
            message = str(exc)
            raise S.VisionUnavailable(message if message.startswith("vision unavailable") else
                                      "vision unavailable: " + message) from None
        finally:
            self.last_used = time.time()

    def close(self):
        self.client.close()

    unload = close

    def restart(self):
        self.client.restart()

    def release(self, manifest):
        """Only this client's own per-request IPC directory is temporary."""
        directory = Path(manifest).parent
        if directory.parent == self.client.cache_dir and directory.name.startswith("request-"):
            shutil.rmtree(directory, ignore_errors=True)

    def shutdown(self):
        self.close()
        self.media.close()
        shutil.rmtree(self.dir, ignore_errors=True)
        if self.log is not None:
            self.log.close()


def mixed_embeddings(manifest, output, max_rows=2**31 - 1):
    """Translate the pinned mtmd helper's linear chunks into common GENI SVE1 records."""
    path = Path(manifest)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "m31-vision-ipc-v1" or value.get("source_pr") != \
            "6b95a9908d2c7b242817a447708aec38e937e2c0" or value.get("n_embd") != 6144 or \
            value.get("llm_inference") is not False or value.get("encoded") is not True:
        raise ValueError("vision unavailable: invalid helper manifest")
    ids, visual = [], 0
    with Path(output).open("xb") as out:
        for index, chunk in enumerate(value["chunks"]):
            count = chunk["rows"]
            if type(count) is not int or count <= 0 or count > max_rows - len(ids) or chunk["index"] != index or \
                    chunk["row"] != len(ids) or chunk["position"] != len(ids) or chunk["n_pos"] != count:
                raise ValueError("vision unavailable: invalid mixed row continuity")
            if chunk["type"] == "text":
                tokens = chunk["tokens"]
                if len(tokens) != count or any(type(token) is not int or not 0 <= token < 2**31 for token in tokens):
                    raise ValueError("vision unavailable: invalid helper text tokens")
                ids.extend(tokens)
            elif chunk["type"] == "image":
                if chunk["visual_row"] != visual:
                    raise ValueError("vision unavailable: invalid visual row continuity")
                file = Path(chunk["embeddings"])
                if file.is_absolute() or file.name != str(file):
                    raise ValueError("vision unavailable: invalid helper embedding filename")
                file = path.parent / file
                if file.is_symlink() or not file.is_file() or file.stat().st_size != count * 6144 * 4:
                    raise ValueError("vision unavailable: invalid helper embedding size")
                # M31 uses ordinary linear decoder positions, so the SVE1 grid
                # carries row count rather than Qwen's multi-axis coordinates.
                out.write(struct.pack("<5i", 0x31455653, count, count, 1, 6144))
                with file.open("rb") as source:
                    shutil.copyfileobj(source, out)
                ids.extend([-1] * count)
                visual += count
            else:
                raise ValueError("vision unavailable: unsupported helper chunk")
    if value["rows"] != len(ids) or value["visual_rows"] != visual or not visual:
        raise ValueError("vision unavailable: invalid helper row totals")
    return ids


def openai_tools(tools):
    # Match common_chat_tools_to_json_oaicompat, including its field order and
    # empty defaults. Extra API keys are not part of the model's tool prompt.
    result = []
    for tool in tools or []:
        function = tool.get("function", tool)
        result.append({"type": "function", "function": {
            "name": function.get("name", ""), "description": function.get("description", ""),
            "parameters": function.get("parameters", {})}})
    return result


def template_messages(messages):
    messages = copy.deepcopy(messages)
    for message in messages:
        calls = message.get("tool_calls") or []
        for i, call in enumerate(calls):
            function = call.get("function", call)
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("MiniMax tool arguments must be JSON objects")
            calls[i] = {**call, "type": "function", "function": {**function, "arguments": arguments}}
    return messages


_BaseTemplate = F.ChatTemplate


class M31Template(_BaseTemplate):
    def __init__(self, path):
        super().__init__(path)
        if P.NS not in self.source:
            raise ValueError("export the MiniMax GGUF's chat_template.jinja first")

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        # The exported template renders recursive tool arguments as mappings;
        # both API adapters normalize calls to the common server's flat form.
        return super().render(template_messages(messages), openai_tools(tools), add_generation_prompt, **kwargs)

    def _detect_caps(self):
        # Qwen's capability probe looks for its <function=NAME> output syntax.
        # MiniMax writes a namespaced <invoke name="NAME"> instead.
        tool = {"name": "m31_capability_probe", "parameters": {"type": "object", "properties": {}}}
        messages = [{"role": "system", "content": "m31_system_probe"}, {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "", "reasoning_content": "m31_reasoning_probe",
                     "tool_calls": [{"function": {"name": tool["name"], "arguments": {}}}]}]
        rendered = self.render(messages, [tool])
        return {"supports_tools": '"name": "m31_capability_probe"' in rendered,
                "supports_tool_calls": P.NS + '<invoke name="m31_capability_probe">' in rendered,
                "supports_system_role": "m31_system_probe" in rendered,
                "supports_parallel_tool_calls": True,
                "supports_preserve_reasoning": "m31_reasoning_probe" in rendered}


class PreparedIds(list):
    def __init__(self, ids, chat_request, cache_prompt=True):
        super().__init__(ids)
        self.chat_request = copy.deepcopy(chat_request)
        self.cache_prompt = cache_prompt


def chat_key(request):
    payload = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


_BaseEngine = S.StrataEngine
_BaseDetokenizer = S.Detokenizer


class M31Detokenizer(_BaseDetokenizer):
    """Retain actual special IDs while using the common UTF-8 incremental decoder."""
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.controls = {token: piece for piece, token in getattr(tokenizer, "special_tokens", {}).items()
                         if piece in P.CONTROL_PIECES}

    def push(self, token):
        text = super().push(token)
        piece = self.controls.get(token)
        controls = {len(text) - len(piece): piece} if piece is not None and text.endswith(piece) else {}
        return P.TokenDelta(text, controls)


class M31Engine(_BaseEngine):
    def __init__(self, exe, args, cwd=None, log=None, env=None, lazy=False, cpus=None, _attempt=0):
        child = dict(os.environ if env is None else env)
        if child.get("GGML_CUDA_DISABLE_FUSION", "1") != "1":
            raise ValueError("MiniMax's accepted native path requires GGML_CUDA_DISABLE_FUSION=1")
        child["GGML_CUDA_DISABLE_FUSION"] = "1"
        super().__init__(exe, args, cwd, log, child, lazy, cpus, _attempt)

    @staticmethod
    def sampling_keys(sampling):
        for name, lower, upper, strict_lower in (
                ("temperature", 0, 100, False), ("top_p", 0, 1, True), ("min_p", 0, 1, False),
                ("repetition_penalty", 0, 100, True), ("frequency_penalty", -100, 100, False),
                ("presence_penalty", -100, 100, False)):
            value = sampling.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    value > upper or value < lower or not math.isfinite(value) or strict_lower and value == lower):
                raise ValueError(f"invalid finite MiniMax sampling value for {name}")
        window = sampling.get("penalty_last_n")
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or not 0 <= window < 2**31):
            raise ValueError("penalty_last_n must be a nonnegative int32")
        cache = sampling.get("cache_prompt", True)
        if not isinstance(cache, bool):
            raise ValueError("cache_prompt must be boolean")
        forwarded = {key: value for key, value in sampling.items() if key not in ("top_k", "seed")}
        result = _BaseEngine.sampling_keys(forwarded) + " cache_prompt=" + str(int(cache))
        if window == 0:
            result = result.replace(" penalty_last_n=64", " penalty_last_n=0")
        top_k = sampling.get("top_k")
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 2**31 - 1:
                raise ValueError("top_k must be a nonnegative int32")
            result += " top_k=" + str(top_k)
        seed = sampling.get("seed")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or not -1 <= seed < 2**64:
                raise ValueError("seed must be uint64 or -1")
            if seed != -1:
                result += " seed=" + str(seed)
        ignore_eos = sampling.get("ignore_eos", False)
        if not isinstance(ignore_eos, bool):
            raise ValueError("ignore_eos must be boolean")
        result += " ignore_eos=" + str(int(ignore_eos))
        chat = sampling.get("m31_chat_b64")
        if chat:
            result += " chat_b64=" + chat
        return result

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        sampling = dict(sampling or {})
        if isinstance(ids, PreparedIds):
            sampling.setdefault("m31_chat_b64", chat_key(ids.chat_request))
            sampling.setdefault("cache_prompt", ids.cache_prompt)
        # A finite deadline also bounds STOP/drain when a parser or grammar
        # raises while the Service still owns the request FIFO.
        if not math.isfinite(float(self.silence_s)) or self.silence_s <= 0:
            raise ValueError("MiniMax requires a finite engine_silence_s greater than zero")
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


_BaseService = S.Service
_engine_silence_s = S.engine_silence_s
_sampling_defaults_from_config = S.sampling_defaults_from_config


def sampling_defaults_from_config(config):
    block = dict(config.get("sampling") or {})
    special = {key: block.pop(key) for key in ("top_k", "seed") if key in block}
    defaults = _sampling_defaults_from_config({**config, "sampling": block})
    # The common engine caps top-k at 64 and treats seed zero as unset. MiniMax's
    # accepted sampler supports an uncapped top-k and the whole uint64 seed.
    M31Engine.sampling_keys({key: value for key, value in special.items() if value is not None})
    defaults.update({key: value for key, value in special.items() if value is not None})
    M31Engine.sampling_keys(defaults)
    return defaults


def engine_silence_s(config):
    seconds = _engine_silence_s(config)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("MiniMax requires a finite engine_silence_s greater than zero")
    return seconds


def validate_config(config):
    vision = config.get("vision")
    if vision is not None and (not isinstance(vision, dict) or not all(
            isinstance(vision.get(key), str) and vision[key] for key in ("exe", "model", "mmproj"))):
        raise ValueError("MiniMax vision requires exe, model and mmproj paths")
    if config.get("replicas"):
        raise ValueError("MiniMax text PR1 uses one native engine process")
    parallel = config.get("parallel", 1)
    if parallel is True or parallel not in (None, False, 0, 1):
        raise ValueError("MiniMax text PR1 supports one request at a time; omit parallel")
    engine_silence_s(config)


class M31Service(_BaseService):
    @property
    def stop_ids(self):
        context = getattr(self, "_m31_request", None)
        return set() if context is not None and getattr(context, "ignore_eos", False) else self._m31_stop_ids

    @stop_ids.setter
    def stop_ids(self, value):
        self._m31_stop_ids = value

    def __init__(self, engine, tokenizer, template, *args, **kwargs):
        self._m31_request = threading.local()
        kwargs["fit_max_tokens"] = True
        super().__init__(engine, tokenizer, template, *args, **kwargs)
        self.sampling_defaults = {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.05,
                                  **self.sampling_defaults}
        eos = tokenizer.encode("[e~[", parse_special=True)
        if isinstance(tokenizer, M31Tokenizer) and len(eos) != 1:
            raise ValueError("MiniMax end-of-generation token is missing")
        self.stop_ids = set(eos)
        self.effort_end = False

    def encode_prompt(self, messages, tools, kwargs):
        # Follow the GGUF template and PR tokenizer oracle exactly. The common
        # Qwen literal-tag substitution changes MiniMax's template token IDs.
        prompt = self.template.render(messages, tools=tools, **kwargs)
        prepared = getattr(self._m31_request, "vision", None)
        if prepared is None:
            return self.tok.encode(prompt, parse_special=True)
        prepared["template_ids"] = self.tok.encode(prompt, parse_special=True)
        if MediaInput.marker in prompt:
            raise ValueError("literal mtmd media marker is reserved in multimodal requests")
        for marker in prepared["markers"]:
            if prompt.count(marker) != 1:
                raise ValueError("MiniMax template did not preserve media markers")
            prompt = prompt.replace(marker, MediaInput.marker)
        cancel = prepared["cancel"]
        with self._turn(cancel):
            if cancel is not None and cancel.is_set():
                raise S.ClientGone()
            manifest = self.vision.encode_prompt(prompt, prepared["media"], prepared["fps"], cancel)
        try:
            if cancel is not None and cancel.is_set():
                raise S.ClientGone()
            try:
                ids = mixed_embeddings(manifest, prepared["path"],
                                       max_rows=self.engine.max_context or getattr(self.engine, "known_ctx", 2**31 - 1))
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                raise S.VisionUnavailable("vision unavailable: " + str(exc)) from None
        finally:
            if hasattr(self.vision, "release"):
                self.vision.release(manifest)
        prepared["rows"] = len(ids)
        return ids

    def prepare(self, messages, tools, kwargs, max_new=None, force=None, req=None, cancel=None):
        request = req or {}
        limit_keys = ("max_output_tokens", "n_predict", "max_completion_tokens", "max_tokens")
        for key in limit_keys:
            value = request.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < -1):
                raise ValueError(f"{key} must be an integer >= -1")
        explicit = next((request[key] for key in limit_keys if request.get(key) is not None), None)
        if explicit is not None:
            max_new = explicit
        if max_new is not None and (isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < -1):
            raise ValueError("generation limit must be an integer >= -1")
        zero = explicit == 0 or req is None and max_new == 0
        cache = request.get("cache_prompt", True)
        if not isinstance(cache, bool):
            raise ValueError("cache_prompt must be boolean")
        has_media = any(isinstance(message.get("content"), list) and any(
                isinstance(part, dict) and part.get("type") not in ("text", "input_text", None)
                for part in message["content"]) for message in messages)
        if request.get("grammar"):
            raise ValueError("raw GBNF grammar is unavailable; use response_format or tool schemas")
        kind, name = F.tool_choice_of(request.get("tool_choice"))
        if kind == "unknown":
            raise ValueError("unsupported MiniMax tool_choice")
        if kind == "named":
            tools = [tool for tool in tools or [] if tool.get("name", (tool.get("function") or {}).get("name")) == name]
            if not tools:
                raise ValueError("tool_choice names no tool in this request")
            kind = "required"
        parallel = request.get("parallel_tool_calls", True)
        if isinstance(request.get("tool_choice"), dict) and "disable_parallel_tool_use" in request["tool_choice"]:
            disabled = request["tool_choice"]["disable_parallel_tool_use"]
            if not isinstance(disabled, bool):
                raise ValueError("disable_parallel_tool_use must be boolean")
            parallel = not disabled
        if not isinstance(parallel, bool):
            raise ValueError("parallel_tool_calls must be boolean")
        if not isinstance(request.get("ignore_eos", False), bool):
            raise ValueError("ignore_eos must be boolean")
        # Common HTTP parsing uses ``value or 0`` and the base Service treats 0
        # as unlimited. Preserve an explicit zero before either interpretation.
        prepared = None
        if has_media:
            if self.vision is None:
                raise S.VisionUnavailable("vision unavailable: configure the opt-in mtmd helper")
            if (getattr(self.engine, "info", {}) or {}).get("vision") == 0:
                raise S.VisionUnavailable("vision unavailable: build the engine with STRATA_M31_VISION=ON")
            messages, media, markers, fps = self.vision.media.prepare(messages, request.get("video_fps"), cancel)
            prepared = {"media": media, "markers": markers, "fps": fps, "cancel": cancel,
                        "path": self.vision.dir / ("request-" + uuid.uuid4().hex + ".sve")}
        self._m31_request.vision = prepared
        try:
            ids, thinking, maximum = super().prepare(messages, tools, kwargs, 1 if zero else max_new,
                                                     force, request, cancel)
        except BaseException:
            if prepared is not None:
                prepared["path"].unlink(missing_ok=True)
            raise
        finally:
            self._m31_request.vision = None
            if prepared is not None:
                self.vision.media.release(prepared["media"])
        if prepared is not None:
            self.embeddings.path = prepared["path"]
            self.embeddings.keys = None
        if zero:
            maximum = 0
        chat = {"messages": template_messages(messages), "tools": openai_tools(tools),
                "tool_choice": kind,
                "parallel_tool_calls": parallel,
                "enable_thinking": kwargs.get("enable_thinking", True),
                "chat_template_kwargs": dict(kwargs)}
        if prepared is not None:
            chat["vision_template_ids"] = prepared["template_ids"]
            chat["vision_prompt_rows"] = prepared["rows"]
        format_ = request.get("response_format") or S.responses_api.text_format(request) or {}
        if format_.get("type") == "json_schema":
            chat["json_schema"] = (format_.get("json_schema") or {}).get("schema", {})
        elif format_.get("type") == "json_object":
            chat["json_schema"] = {"type": "object"}
        if force and not thinking:
            chat["assistant_prefix"] = force
        return PreparedIds(ids, chat, cache), thinking, maximum

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None):
        if isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < 0:
            raise ValueError("prepared generation limit must be a nonnegative integer")
        if max_new == 0:
            # A zero completion does not acquire the native worker or read a
            # prompt. It cannot turn into an unbounded request on any backend.
            yield "done", {"finish": "length", "completion_tokens": 0, "reused": 0, "timings": None,
                           "reasoning_tokens": 0, "reasoning_recoveries": 0}
            return
        sampling = dict(sampling or {})
        if isinstance(ids, PreparedIds):
            sampling["m31_chat_b64"] = chat_key(ids.chat_request)
            sampling["cache_prompt"] = ids.cache_prompt
        ignore_eos = sampling.get("ignore_eos", False)
        if not isinstance(ignore_eos, bool):
            raise ValueError("ignore_eos must be boolean")
        previous = getattr(self._m31_request, "ignore_eos", False)
        self._m31_request.ignore_eos = ignore_eos
        try:
            yield from super().run(ids, thinking, tools, max_new, sampling, cancel, force)
        finally:
            self._m31_request.ignore_eos = previous


def install():
    """Register the adapter in this server process; shared source files stay unchanged."""
    ST.Tokenizer = M31Tokenizer
    F.ChatTemplate = S.ChatTemplate = M31Template
    F.OutputParser = S.OutputParser = P.M31OutputParser
    F.forced_call = S.forced_call = P.forced_call
    F.THINK_END = S.THINK_END = P.THINK_END
    S.IM_END = "[e~["
    S.REASONING_WRAP_UP = S.REASONING_WRAP_UP.replace("</think>", P.THINK_END)
    S.StrataEngine = M31Engine
    S.Detokenizer = M31Detokenizer
    S.Service = M31Service
    S.Vision = M31Vision
    install_video_reader()
    S.engine_silence_s = engine_silence_s
    S.sampling_defaults_from_config = sampling_defaults_from_config


def main():
    install()
    for i, value in enumerate(sys.argv[:-1]):
        if value in ("--config", "--tokenizer", "--mcp-config"):
            sys.argv[i + 1] = os.path.abspath(sys.argv[i + 1])
        if value == "--config":
            validate_config(json.loads(Path(sys.argv[i + 1]).read_text(encoding="utf-8")))
    for i, value in enumerate(sys.argv):
        for option in ("--config", "--tokenizer", "--mcp-config"):
            if value.startswith(option + "="):
                path = os.path.abspath(value[len(option) + 1:])
                sys.argv[i] = option + "=" + path
                if option == "--config":
                    validate_config(json.loads(Path(path).read_text(encoding="utf-8")))
    os.chdir(ROOT)
    return S.main()


if __name__ == "__main__":
    sys.exit(main())
