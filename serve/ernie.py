"""serve/ernie.py - ERNIE-5.5-Flash's turn format for the server (GGUF arch `ernie5_5`).

ERNIE writes an assistant turn as (Baidu's encoding_ernie.py / decoding_ernie.py, vendored in serve/ernie_encoding.py):

    <ernie:think>{chain of thought}</ernie:think>{the answer}
    <ernie:tool_call>
    <invoke name="NAME"><parameter name="KEY">VALUE</parameter>...</invoke>
    </ernie:tool_call><|ernie:im_end|>

The generation prompt ends with `<ernie:think>` (thinking on), so the reply starts inside the reasoning; with thinking
off it ends with `<ernie:think></ernie:think>` and the reply is the answer.  A string parameter is written raw, every
other value as JSON, which is the Qwen form's own convention.  `<|ernie:im_end|>` ends the turn.

The server's OutputParser (serve/frontend.py) reads the Qwen form: reasoning, `</think>`, the answer, and calls as
`<tool_call><function=NAME><parameter=P>...`.  ErnieTranslator rewrites ERNIE's markers into that form as they stream
(holding back only a marker still arriving, and a call block until it closes), and ErnieOutputParser is the
OutputParser behind it, as serve/inkling.py and serve/dsml.py do for their families.

render() is Baidu's own renderer (serve/ernie_encoding.py); serve/test_ernie.py checks the prompts it gives.
"""
from __future__ import annotations

import json
import re

from serve import ernie_encoding as enc
from serve.frontend import CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START, OutputParser

MASK = enc.MASK
START = enc.START
END = enc.END                                   # <|ernie:im_end|>: the end of the turn (a stop token)
THINK = enc.THINK
THINK_END = enc.THINK_END
CALL = enc.CALL
CALL_END = enc.CALL_END

_MARKERS = (THINK_END, CALL, CALL_END, END, MASK, THINK, START)
THINK_CLOSE = "</think>\n\n"                    # the Qwen form's reasoning end, as the other family modules write it

# What the server writes into the engine's input in ERNIE's spelling (the Qwen spellings are in server.py)
REASONING_CLOSE = THINK_END
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n" + THINK_END
ANSWER_OPEN = None                              # thinking off is the template's own empty think block

_INVOKE = re.compile(r'<invoke name="([^"]*)">')
_PARAM = re.compile(r'<parameter name="([^"]*)">')

# ------------------------------------------------------------------------------------------------ render
_EFFORT = {"low": "low", "minimal": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


def effort_of(kwargs: dict):
    """The request's template kwargs -> (enable_thinking, reasoning_effort) for Baidu's renderer."""
    if kwargs.get("enable_thinking") is False:
        return False, enc.MISSING
    e = kwargs.get("reasoning_effort")
    if e is None or e == "":
        return None, enc.MISSING                # the template's default: high
    key = str(e).strip().lower()
    if key in ("none", "off", "false", "disabled"):
        return False, enc.MISSING
    return None, _EFFORT.get(key, "high")


def _messages(messages):
    """The server's messages -> the renderer's: a tool call's arguments as a parsed object or its JSON string (both
    accepted), the reasoning under `reasoning_content`; anything else is passed as it is."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError(f"Unknown message: {m!r}")
        m = dict(m)
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def render(messages, tools=None, kwargs=None) -> str:
    """The prompt text, with the generation prompt (Baidu's render_chat, byte for byte)."""
    kwargs = kwargs or {}
    thinking, effort = effort_of(kwargs)
    tool_list = [t for t in (tools or []) if isinstance(t, dict)] or None
    options = {}
    if kwargs.get("parallel_tool_calls") is False:
        options["parallel_tool_calls"] = False
    return enc.render_chat(_messages(messages), tool_list, add_generation_prompt=True, enable_thinking=thinking,
                           reasoning_effort=effort, options=options or None).text


def thinking_on(kwargs: dict) -> bool:
    return effort_of(kwargs or {})[0] is not False


# ------------------------------------------------------------------------------------------------ output
class ErnieTranslator:
    """ERNIE's markers -> the Qwen form, a delta at a time.

    thinking: the prompt ended inside <ernie:think>, so the reply starts with the reasoning; otherwise the reply is
    the answer from the first token."""

    def __init__(self, thinking: bool = True):
        self.closed = not thinking             # `</think>` written (or never needed)
        self.buf = ""                          # text held back: a marker still arriving
        self.mode = "think" if thinking else "text"   # think | text | tool
        self.call: list[str] = []              # the open call block's text
        self.opened: set[str] = set()          # tool names a call block named
        self.emitted = False

    @property
    def held(self) -> str:
        return self.buf

    @property
    def clean(self) -> bool:
        """No marker and no call block is held: a point the server may write into."""
        return not self.buf and self.mode != "tool"

    def feed(self, delta: str) -> str:
        self.buf += delta
        return self._run(final=False)

    def flush(self) -> str:
        """The end of the output: a call block cut before its close keeps the calls that closed in it."""
        out = self._run(final=True)
        if self.mode == "tool":
            out += self._emit_calls()
            self.mode = "text"
        return out

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
            if self.mode != "tool" and any(m.startswith(self.buf) for m in _MARKERS):
                if final:
                    self._text(self.buf, out)
                    self.buf = ""
                break                          # it may still become a marker
            if self.mode == "tool" and CALL_END.startswith(self.buf) and not final:
                break
            self._text("<", out)               # an ordinary '<' (inside a call block: part of its text)
            self.buf = self.buf[1:]
        return "".join(out)

    def _close(self, out: list[str]) -> None:
        if not self.closed:
            out.append(THINK_CLOSE)
            self.closed = True

    def _text(self, text: str, out: list[str]) -> None:
        if not text:
            return
        if self.mode == "tool":
            self.call.append(text)
            return
        if self.mode == "think" and not self.closed:
            out.append(text)
            self.emitted = True
            return
        self._close(out)
        out.append(text)
        self.emitted = True

    def _marker(self, marker: str, out: list[str]) -> None:
        if marker == THINK_END:
            if self.mode == "tool":
                self.call.append(marker)
                return
            self._close(out)
            self.mode = "text"
        elif marker == CALL:
            if self.mode != "tool":
                self._close(out)
                self.mode, self.call = "tool", []
        elif marker == CALL_END:
            if self.mode == "tool":
                out.append(self._emit_calls())
                self.mode = "text"
        # END / MASK end the turn (stop tokens: normally never in the text); a stray THINK / START is dropped

    def _emit_calls(self) -> str:
        """The call block's <invoke> elements -> the Qwen call text, one <tool_call> each.  An invoke that never
        closed (a cut reply) is dropped."""
        text, self.call = "".join(self.call), []
        out = ""
        i = 0
        while True:
            m = _INVOKE.search(text, i)
            if not m:
                break
            end = text.find("</invoke>", m.end())
            if end < 0:
                break
            name, body = m.group(1).strip(), text[m.end():end]
            i = end + len("</invoke>")
            if not name:
                continue
            params = []
            j = 0
            while True:
                p = _PARAM.search(body, j)
                if not p:
                    break
                pe = body.find("</parameter>", p.end())
                if pe < 0:
                    break
                params.append((p.group(1), body[p.end():pe]))
                j = pe + len("</parameter>")
            call = ("\n\n" if self.emitted else "") + CALL_START + "\n" + FUNC_START + name + ">\n"
            for key, value in params:
                call += PARAM_START + key + ">\n" + value + "\n" + PARAM_END + "\n"
            call += FUNC_END + "\n</tool_call>\n"
            out += call
            self.opened.add(name)
            self.emitted = True
        return out


class ErnieOutputParser(OutputParser):
    """OutputParser for ERNIE: its markers are read as the template-form text they are."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ernie = ErnieTranslator(thinking=self.state == "reasoning")

    def _callable(self, name: str) -> bool:
        """A call the model opened with <ernie:tool_call> is a call, declared or not (as serve/inkling.py does)."""
        return super()._callable(name) or name in self.ernie.opened

    def feed(self, delta: str):
        return super().feed(self.ernie.feed(delta))

    def finish(self, reason: str | None = None):
        if reason not in (None, "stop") and self.ernie.held:
            self.ernie.buf = ""                # cut inside a marker: not text the model meant
        rest = self.ernie.flush()
        out = super().feed(rest) if rest else []
        out += super().finish(reason)
        return out

    @property
    def clean(self) -> bool:
        return not self.buf and self.ernie.clean


def to_ernie_force(force: str | None) -> str | None:
    """forced_call's opening of a call in the Qwen form -> the same opening in ERNIE's spelling (the call block and
    the invoke tag; the model writes the parameters and the close)."""
    if not force:
        return force
    at = force.find(FUNC_START)
    name = force[at + len(FUNC_START):] if at >= 0 else ""
    if name.endswith(">\n"):
        return CALL + "\n" + '<invoke name="' + name[:-2] + '">'
    return CALL + "\n"


__all__ = ["ErnieTranslator", "ErnieOutputParser", "render", "effort_of", "thinking_on", "to_ernie_force",
           "REASONING_CLOSE", "REASONING_WRAP_UP", "ANSWER_OPEN", "END"]
