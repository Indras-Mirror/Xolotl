"""serve/inkling.py - Inkling-2-Mini's turn format (message blocks, JSON tool calls) for the server.

Inkling 2 (GGUF arch `inkling2_loop`, or `inkling2` without the loop) writes an assistant turn as a list of blocks,
each closed by `<|end_message|>` (its chat template, llama.cpp `common/parsers/inkling.cpp`):

    <|message_model|><|content_thinking|>{chain of thought}<|end_message|>
    <|message_model|><|content_text|>{the answer}<|end_message|>
    <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>

`<|content_model_end_sampling|>` (200006, eos) ends the turn; `<|end_message|>` (200010) ends one message.  The
template has NO generation prompt: the model writes the `<|message_model|>` header itself, so a prompt ends after
the last message's `<|end_message|>`.  Thinking effort is a template variable (0.0 .. 0.99, or none/minimal/low/
medium/high/xhigh/max), written into the prompt as a short system message.

The server's OutputParser (serve/frontend.py) reads the Qwen form - reasoning, `</think>`, the answer, and calls as
`<tool_call><function=NAME><parameter=P>...`.  InklingTranslator rewrites Inkling's blocks into that form as they
stream (holding back only a marker that is still arriving), and InklingOutputParser is the OutputParser behind it, as
serve/dsml.py and serve/atem.py do for their families.  So tool_choice, streamed arguments, the reasoning/answer
split and the recovery rules behave for Inkling as for the others.

render() is the model's own chat template in Python (the pack's chat_template.jinja is the source of truth): the
tests render the official template with jinja2 and require the same bytes.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

from serve.frontend import CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START, OutputParser

HEADER = "<|message_model|>"
THINK = "<|content_thinking|>"
TEXT = "<|content_text|>"
TOOL_JSON = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
EOS = "<|content_model_end_sampling|>"          # 200006: one token, the end of the turn
EOS_ID = 200006

ROLE_TOKEN = {"user": "<|message_user|>", "assistant": HEADER,
              "system": "<|message_system|>", "tool": "<|message_tool|>"}

# Tokens that carry no text of their own.  EOS is normally a stop token (never in the output); the markers of other
# messages are never generated, but a stray one is dropped rather than shown.
_DROP = (EOS, "<|endoftext|>", "<|begin_of_text|>", "<|message_user|>", "<|message_system|>",
         "<|message_tool|>", "<|content_tool_error|>", "<|content_invoke_tool_text|>", "<|content_xml|>",
         "<|content_image|>", "<|unused_200054|>")
_MARKERS = (HEADER, THINK, TEXT, TOOL_JSON, END) + _DROP

THINK_CLOSE = "</think>\n\n"                    # the Qwen form's reasoning end, as the other family modules write it
_NAME_MAX = 200                                 # a header followed by more than this and no marker is text

# What the server writes into the engine's input in Inkling's spelling (its Qwen spellings are in server.py).
# The model writes the header itself; thinking off gets the answer's opening appended so the reply is the answer.
ANSWER_OPEN = HEADER + TEXT
REASONING_CLOSE = "\n" + END                    # close the thinking block; the model opens the next one
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n" + END


# ------------------------------------------------------------------------------------------------ render
_EFFORT = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}


def effort_value(reasoning_effort) -> float:
    """The template's reasoning_effort -> its number: a name from the map, else a float in [0.0, 0.99]."""
    if reasoning_effort is None:
        return 0.9
    if isinstance(reasoning_effort, str):
        key = reasoning_effort.strip()
        if key not in _EFFORT:
            raise ValueError(f"Unknown reasoning_effort: {reasoning_effort}")
        return _EFFORT[key]
    num = float(reasoning_effort)
    if not 0.0 <= num <= 0.99:
        raise ValueError("reasoning_effort must be in [0.0, 0.99]")
    return num


def effort_from_kwargs(kwargs: dict):
    """The request's template kwargs -> render()'s reasoning_effort (thinking off is the template's 0)."""
    if kwargs.get("enable_thinking") is False:
        return 0.0
    return kwargs.get("reasoning_effort")


def _effort_block(reasoning_effort) -> str:
    num = effort_value(reasoning_effort)
    text = "0" if num == 0.0 else str(num)
    return f"<|message_system|><|content_text|>Thinking effort level: {text}<|end_message|>"


def _jd(value, **kwargs) -> str:
    return json.dumps(value, ensure_ascii=False, **kwargs)


def _field(obj, name):
    return obj.get(name) if isinstance(obj, Mapping) else None


def _string(value) -> str:
    """Jinja renders a truthy message name / reasoning content as its string form."""
    return value if isinstance(value, str) else str(value)


def render(messages, tools=None, reasoning_effort=None) -> str:
    """The official Inkling chat template (`chat_template.jinja`), byte for byte.

    `messages` are the template's dicts: role/content, an assistant's `reasoning_content` and
    `tool_calls` (each `{"id", "function": {"name", "arguments": {parsed}}}`), and a tool result's `name` or
    `tool_call_id`.  Tool call arguments must be parsed objects, as the template requires."""
    out: list[str] = []
    if tools:
        specs = []
        for tool in tools:
            fn = tool["function"] if isinstance(tool, Mapping) and "function" in tool else tool
            specs.append({
                "description": (_field(fn, "description") or "") if isinstance(fn, Mapping) else "",
                "name": _field(fn, "name"),
                "parameters": (_field(fn, "parameters") or {}) if isinstance(fn, Mapping) else {},
                "type": (_field(tool, "type") or "function") if isinstance(tool, Mapping) else "function",
            })
        out.append("<|message_system|>tool_declare<|content_xml|>"
                   + _jd(specs, sort_keys=True, separators=(",", ":")) + "<|end_message|>")

    effort_emitted = False

    def emit_effort():
        nonlocal effort_emitted
        out.append(_effort_block(reasoning_effort))
        effort_emitted = True

    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError(f"Unknown message role: {message!r}")
        role = message.get("role")
        if role not in ROLE_TOKEN:
            raise ValueError(f"Unknown message role: {role}")
        if not effort_emitted and role != "system":
            emit_effort()

        header = ROLE_TOKEN[role]
        if role == "tool":
            name = message.get("name")
            if name:
                header += _string(name)
            else:
                cid = message.get("tool_call_id")
                if cid:
                    for prev in messages:
                        if not isinstance(prev, Mapping) or prev.get("role") != "assistant" or not prev.get("tool_calls"):
                            continue
                        for tc in prev["tool_calls"]:
                            fn = _field(tc, "function")
                            if isinstance(tc, Mapping) and "id" in tc and tc.get("id") == cid \
                                    and isinstance(fn, Mapping) and "name" in fn:
                                header += _string(fn["name"])
        elif role == "assistant" and message.get("name"):
            header += _string(message["name"])

        rtok = header
        reasoning = message.get("reasoning_content")
        if role == "assistant" and reasoning:
            out.append(rtok + THINK + _string(reasoning) + END)

        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if isinstance(content, str) and (content or not (role == "assistant" and (tool_calls or reasoning))):
            out.append(rtok + TEXT + content + END)
        elif content:
            for part in content:
                if isinstance(part, str):
                    out.append(rtok + TEXT + part + END)
                    continue
                kind = _field(part, "type")
                if "type" not in part or kind in ("text", "input_text"):
                    text = _field(part, "text")
                    out.append(rtok + TEXT + (text if isinstance(text, str) else "") + END)
                elif kind in ("image", "input_image", "image_url"):
                    if role not in ("user", "tool"):
                        raise ValueError("Image content is only supported for user and tool messages")
                    out.append(rtok + "<|content_image|><|unused_200054|>" + END)
                elif kind in ("audio", "input_audio", "audio_url"):
                    raise ValueError("Inkling 2 does not accept audio input")
                else:
                    raise ValueError(f"Unsupported content part type: {kind}")

        if role == "assistant" and tool_calls:
            for tc in tool_calls:
                fn = _field(tc, "function")
                name = _field(fn, "name")
                if not isinstance(name, str):
                    raise ValueError("tool call function name must be a string")
                if message.get("name") and message["name"] != name:
                    raise ValueError(f"assistant name {message['name']} does not match its tool call {name}")
                args = _field(fn, "arguments") if _field(fn, "arguments") else {}
                if isinstance(args, str):
                    raise ValueError("tool call arguments must be a parsed object, not a JSON string; "
                                     "canonicalize upstream")
                if not isinstance(args, Mapping):
                    raise ValueError("tool call arguments must be an object")
                pairs = []
                for key, value in sorted(args.items(), key=lambda kv: kv[0]):
                    pairs.append(_jd(key) + ":" + _jd(value, separators=(",", ":")))
                out.append(HEADER + name + TOOL_JSON + '{"name":' + _jd(name, sort_keys=True, separators=(",", ":"))
                           + ',"args":{' + ",".join(pairs) + "}}" + END)

        if role == "assistant":
            out.append(EOS)

    if not effort_emitted:
        emit_effort()
    return "".join(out)


# ------------------------------------------------------------------------------------------------ output
def _value_text(value) -> str:
    """A JSON argument -> the Qwen parameter text: a string as it is, every other type as compact JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class InklingTranslator:
    """Inkling's blocks -> the Qwen form, a delta at a time.

    thinking: the prompt let the model think (effort above 0), so the reply's first block is the reasoning; otherwise
    (the prompt ended at ANSWER_OPEN) the reply is the answer from the first token."""

    def __init__(self, thinking: bool = True):
        self.thinking = thinking
        self.closed = not thinking             # `</think>` written (or never needed)
        self.buf = ""                          # text held back: a marker still arriving
        self.mode = "think" if thinking else "text"   # think | name | text | tool
        self.in_block = False
        self.held_name = ""                    # text after a <|message_model|> header (a tool's name)
        self.tool_name = None
        self.tool_json: list[str] = []
        self.opened: set[str] = set()          # tool names an invoke block opened
        self.emitted = False                   # the client has seen translated text

    @property
    def held(self) -> str:
        """Text held back because it may be the start of a marker."""
        return self.buf

    @property
    def clean(self) -> bool:
        """No marker and no name/JSON is held: a point the server may write into (a force, a thinking close).

        An open thinking block is clean (the thinking budget closes it there, as the template would); an open answer
        block is not (a forced call waits for its <|end_message|>, so the server never writes into its text)."""
        if self.buf or self.mode in ("name", "tool"):
            return False
        return not self.in_block or not self.closed

    def feed(self, delta: str) -> str:
        self.buf += delta
        return self._run(final=False)

    def flush(self) -> str:
        """The end of the output: a complete call block is still a call; an unfinished one is dropped."""
        out = self._run(final=True)
        if self.mode == "tool":
            out += self._emit_call()
        self.tool_name, self.tool_json, self.held_name = None, [], ""
        return out

    # ------------------------------------------------------------------------------------
    def _run(self, final: bool) -> str:
        out: list[str] = []
        while self.buf:
            lt = self.buf.find("<")
            if lt < 0:
                self._text(self.buf, out)
                self.buf = ""
                break
            if lt:
                self._text(self.buf[:lt], out)
                self.buf = self.buf[lt:]
            marker = next((m for m in _MARKERS if self.buf.startswith(m)), None)
            if marker is not None:
                self.buf = self.buf[len(marker):]
                self._marker(marker, out)
                continue
            if any(m.startswith(self.buf) for m in _MARKERS):
                if final:
                    self._text(self.buf, out)
                    self.buf = ""
                break                          # it may still become a marker
            self._text("<", out)               # an ordinary '<'
            self.buf = self.buf[1:]
        return "".join(out)

    def _close(self, out: list[str]) -> None:
        if not self.closed:
            out.append(THINK_CLOSE)
            self.closed = True

    def _text(self, text: str, out: list[str]) -> None:
        if not text:
            return
        if self.mode == "name":
            self.held_name += text
            if len(self.held_name) > _NAME_MAX:        # a header followed by prose, not a call
                chunk, self.held_name = self.held_name, ""
                self.mode = "text" if self.closed else "think"
                self._text(chunk, out)
            return
        if self.mode == "tool":
            self.tool_json.append(text)
            return
        if self.mode == "think" and not self.closed:
            out.append(text)
            self.emitted = True
            return
        self._close(out)
        out.append(text)
        self.emitted = True

    def _marker(self, marker: str, out: list[str]) -> None:
        if marker in _DROP:
            return
        if marker == HEADER:
            self._finish_block(out)
            self.mode, self.held_name = "name", ""
            return
        if marker == THINK:
            self._finish_block(out)
            self.held_name = ""
            self.mode = "think" if not self.closed else "text"
            self.in_block = True
            return
        if marker == TEXT:
            self._finish_block(out)
            self.held_name = ""
            self._close(out)
            self.mode, self.in_block = "text", True
            return
        if marker == TOOL_JSON:
            self._finish_block(out)
            self._close(out)
            self.tool_name, self.tool_json = self.held_name.strip(), []
            self.held_name = ""
            self.mode, self.in_block = "tool", True
            return
        if marker == END:
            self._finish_block(out)
            if self.mode == "think" and not self.closed:
                self._close(out)               # the reasoning block ended: the thinking is over
            self.mode, self.in_block = ("text" if self.closed else "think"), False

    def _finish_block(self, out: list[str]) -> None:
        if self.mode == "tool":
            out.append(self._emit_call())
            self.mode = "text" if self.closed else "think"
        self.in_block = False

    def _emit_call(self) -> str:
        """The buffered JSON arguments -> the Qwen call text (the header name names the call)."""
        raw, name = "".join(self.tool_json), self.tool_name
        self.tool_json, self.tool_name = [], None
        if not raw.strip():
            return ""
        try:
            body = json.loads(raw)
        except ValueError:
            return ""                          # malformed or cut: no call the model meant
        if not isinstance(body, dict):
            return ""
        if not name:
            name = body.get("name") if isinstance(body.get("name"), str) else ""
        args = body.get("args")
        if hasattr(args, "keys"):
            args = dict(args)
        if not isinstance(name, str) or not name.strip() or not isinstance(args, dict):
            return ""
        name = name.strip()
        text = ("\n\n" if self.emitted else "") + CALL_START + "\n" + FUNC_START + name + ">\n"
        for key, value in args.items():
            text += PARAM_START + str(key) + ">\n" + _value_text(value) + "\n" + PARAM_END + "\n"
        text += FUNC_END + "\n</tool_call>\n"
        self.opened.add(name)
        self.emitted = True
        return text


class InklingOutputParser(OutputParser):
    """OutputParser for Inkling: its blocks are read as the template-form text they are."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ink = InklingTranslator(thinking=self.state == "reasoning")

    def _callable(self, name: str) -> bool:
        """A call the model opened with <|content_invoke_tool_json|> is a call, declared or not (the client can
        answer it with an error; as text it would read as the answer) - as serve/dsml.py does for DSML calls."""
        return super()._callable(name) or name in self.ink.opened

    def feed(self, delta: str):
        return super().feed(self.ink.feed(delta))

    def finish(self, reason: str | None = None):
        if reason not in (None, "stop") and self.ink.held:
            self.ink.buf = ""                  # cut inside a marker: not text the model meant
        rest = self.ink.flush()
        out = super().feed(rest) if rest else []
        out += super().finish(reason)
        return out

    @property
    def clean(self) -> bool:
        return not self.buf and self.ink.clean


def to_inkling_force(force: str | None) -> str | None:
    """forced_call's opening of a call in the Qwen form -> the same opening in Inkling's spelling (the model's own
    header, the tool's name, and the JSON object's opening; the model writes the arguments and the close)."""
    if not force:
        return force
    at = force.find(FUNC_START)
    name = force[at + len(FUNC_START):] if at >= 0 else ""
    if name.endswith(">\n"):
        name = name[:-2]
        return HEADER + name + TOOL_JSON + '{"name":' + _jd(name) + ',"args":{'
    return HEADER


__all__ = ["InklingTranslator", "InklingOutputParser", "render", "effort_value", "effort_from_kwargs",
           "to_inkling_force", "ANSWER_OPEN", "REASONING_CLOSE", "REASONING_WRAP_UP", "EOS_ID"]
