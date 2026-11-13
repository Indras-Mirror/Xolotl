"""serve/atem.py - Muse Glimmer's turn format (ATEM tool calls, message channels) for the server.

Muse Glimmer (Meta, `muse-glimmer` GGUF arch) writes an assistant turn as one or more messages after the prompt's
`<|start|>assistant` (llama.cpp common/parsers/muse-glimmer.cpp, the model's chat template):

     to=self<|message|>{chain of thought}<|eom|>
    <|start|>assistant to=user<|message|>{the answer}<|eot|>

and a tool call as a message to the tool, in ATEM markup:

    <|start|>assistant to=get_weather<|message|><atem:function_calls>
    <atem:invoke name="get_weather">
    <atem:parameter name="city">Melbourne</atem:parameter>
    </atem:invoke>
    </atem:function_calls><|eot|>

`<|eom|>` ends a message with more to come (the reasoning, a call followed by another); `<|eot|>` ends the turn (a stop
token, never in the text).  Parameter values are written as they are (strings raw, other types as JSON).

The server's OutputParser (serve/frontend.py) reads the Qwen form - reasoning, `</think>`, the answer, and calls as
`<tool_call><function=NAME><parameter=P>...`.  AtemTranslator rewrites Glimmer's text into that form as it streams
(holding back only a marker or tag that is still arriving), and AtemOutputParser is the OutputParser behind it - as
serve/dsml.py does for DeepSeek-V4.1.  So tool_choice, streamed arguments, the reasoning/answer split and the
recovery rules behave for Glimmer as for the other families.
"""
from __future__ import annotations

import re

from serve.frontend import CALL_START, FUNC_START, Event, OutputParser

START, MESSAGE, EOM, EOT, EOS = "<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "<|end_of_text|>"
SPECIALS = (START, MESSAGE, EOM, EOT, EOS)
THINK_CLOSE = "</think>\n\n"
_HEADER_MAX = 400               # a header (`<|start|>assistant to=functions.x`) longer than this is text
_TAG_MAX = 600                  # an ATEM tag longer than this is text

_RECIPIENT = re.compile(r"\bto=(?P<to>[^\s<]+)")
_CALLS = re.compile(r"<atem:function_calls\s*>")
_CALLS_END = re.compile(r"</atem:function_calls\s*>")
_INVOKE = re.compile(r'<atem:invoke\s+name\s*=\s*"(?P<name>[^"]*)"\s*>')
_INVOKE_END = re.compile(r"</atem:invoke\s*>")
_PARAM = re.compile(r'<atem:parameter\s+name\s*=\s*"(?P<name>[^"]*)"\s*>')
_PARAM_END = re.compile(r"</atem:parameter\s*>")
_ATEM_WORDS = ("<atem:function_calls", "</atem:function_calls", "<atem:invoke", "</atem:invoke",
               "<atem:parameter", "</atem:parameter")

# what the server writes into the engine's input in Glimmer's spelling (its Qwen spellings are in server.py)
REASONING_CLOSE = EOM + START + "assistant"
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer." + EOM + START + "assistant"
ANSWER_OPEN = " to=user" + MESSAGE      # thinking off: the prompt ends in the answer's header


def _special_prefix(s: str) -> bool:
    """`s` (starting with '<') may still become one of the special markers."""
    return any(m.startswith(s) for m in SPECIALS)


def _atem_prefix(s: str) -> bool:
    """`s` (starting with '<', no '>' yet) may still become an ATEM tag."""
    if len(s) > _TAG_MAX or "\n" in s:
        return False
    return any(w.startswith(s) or s.startswith(w) for w in _ATEM_WORDS)


class AtemTranslator:
    """Glimmer's turn text -> the Qwen form, a delta at a time.

    thinking: the parser behind starts in the reasoning (the prompt ended at `<|start|>assistant`); otherwise the
    prompt ended in the answer's header (ANSWER_OPEN) and the text is the answer from the first token."""

    def __init__(self, thinking: bool = True, on_open=None):
        self.buf = ""
        self.closed = not thinking        # `</think>` written (or never needed)
        self.state = "header" if thinking else "user"   # header | self | user | tool (+ calls/invoke/param inside tool)
        self.tool_state = "out"           # out | calls | invoke | param
        self.tool = None
        self.prev = None                  # the channel of the message that just ended
        self.on_open = on_open            # called with each tool name an ATEM invoke opened

    @property
    def held(self) -> str:
        return self.buf

    def feed(self, delta: str) -> str:
        self.buf += delta
        return self._run(final=False)

    def flush(self) -> str:
        out = self._run(final=True)
        if self.state == "tool" and self.tool_state in ("invoke", "param"):
            # the turn ended inside a call (no closing tags): close it so the parser can judge it
            out += ("\n</parameter>\n" if self.tool_state == "param" else "") + "</function>\n</tool_call>\n"
            self.tool_state = "out"
        return out

    # ------------------------------------------------------------------------------------------------
    def _close(self, out: list[str]) -> None:
        if not self.closed:
            out.append(THINK_CLOSE)
            self.closed = True

    def _run(self, final: bool) -> str:
        out: list[str] = []
        while self.buf:
            if self.state == "header":
                if not self._header(out, final):
                    break
                continue
            lt = self.buf.find("<")
            if lt < 0:
                self._text(self.buf, out)
                self.buf = ""
                break
            if lt:
                self._text(self.buf[:lt], out)
                self.buf = self.buf[lt:]
            got = self._marker(out)
            if got == "wait":
                if final:
                    self._text(self.buf, out)
                    self.buf = ""
                break
            if got is None:                     # an ordinary '<'
                self._text("<", out)
                self.buf = self.buf[1:]
        return "".join(out)

    def _header(self, out: list[str], final: bool) -> bool:
        """Read `[<|start|>assistant][ to=R]<|message|>`; False when it is still arriving."""
        at = self.buf.find(MESSAGE)
        if at < 0:
            if not final and len(self.buf) <= _HEADER_MAX and (self.buf.find("<") < 0 or self._header_may_continue()):
                return False
            # `to=NAME` then the markup with no <|message|> (the form of the system prompt's own example)
            m = re.match(r"\s*(?:<\|start\|>)?\s*(?:assistant)?\s*to=(?P<to>[^\s<]+)\s*(?=<atem:)", self.buf)
            if m and m.group("to") not in ("self", "user"):
                self.buf = self.buf[m.end():]
                self._close(out)
                self.sep = self.prev == "user"
                self.state, self.tool_state, self.tool = "tool", "out", m.group("to")
                return True
            if not final and re.match(r"\s*(?:<\|start\|>)?\s*(?:assistant)?\s*to=[^\s<]+\s*(?:<(?:a(?:t(?:e(?:m:?)?)?)?)?)?$", self.buf) and len(self.buf) <= _HEADER_MAX:
                return False                    # `to=NAME\n\n` - the markup may follow
            # no header: the model wrote text where one belongs - it is the current channel's text
            text, self.buf = self.buf, ""
            for m in SPECIALS:
                text = text.replace(m, "")
            self.state = "self" if not self.closed else "user"
            self._text(text, out)
            return True
        head, self.buf = self.buf[:at], self.buf[at + len(MESSAGE):]
        m = _RECIPIENT.search(head)
        to = m.group("to") if m else "user"
        if to == "self":
            if self.closed:                     # more thinking after the answer began: shown as text
                self.state = "user"
            else:
                self.state = "self"
        elif to == "user":
            self._close(out)
            self.state = "user"
        else:
            self._close(out)
            self.sep = self.prev == "user"      # a call after answer text starts on a line of its own
            self.state, self.tool_state, self.tool = "tool", "out", to
        return True

    def _header_may_continue(self) -> bool:
        s = self.buf
        # only specials, `assistant`, ` to=NAME` and spaces belong in a header
        rest = s.replace(START, "").replace(EOM, "")
        last = rest.rfind("<")
        if last >= 0 and not _special_prefix(rest[last:]):
            return False
        if last >= 0:
            rest = rest[:last]
        rest = rest.lstrip()
        if "assistant".startswith(rest):
            return True
        if rest.startswith("assistant"):
            rest = rest[len("assistant"):]
        rest = rest.lstrip()
        return "to=".startswith(rest) or re.fullmatch(r"to=[^\s<]*\s*", rest) is not None

    def _text(self, text: str, out: list[str]) -> None:
        if not text:
            return
        if self.state == "tool" and self.tool_state in ("out", "calls", "invoke"):
            if text.strip() == "":
                return                          # the layout between tags
        out.append(text)

    def _end_message(self, out: list[str]) -> None:
        if self.state == "tool" and self.tool_state in ("invoke", "param"):
            out.append(("\n</parameter>\n" if self.tool_state == "param" else "") + "</function>\n</tool_call>\n")
        self.tool_state = "out"
        self.prev, self.state = self.state, "header"

    def _marker(self, outl: list[str]):
        """The marker / tag at the start of self.buf: True (consumed), "wait" (may still become one) or None (text)."""
        s = self.buf
        for m in (EOM, EOT, EOS, START, MESSAGE):
            if s.startswith(m):
                self.buf = s[len(m):]
                if m == EOM:
                    self._end_message(outl)
                elif m == START:                # a message start without the <|eom|> before it
                    self._end_message(outl)
                    self.buf = START + self.buf
                    self.state = "header"
                # <|eot|> / <|end_of_text|> are stop tokens (never fed); a stray <|message|> is dropped
                return True
        if _special_prefix(s):
            return "wait"
        if self.state != "tool":
            return None
        gt = s.find(">")
        if gt < 0:
            return "wait" if _atem_prefix(s) else None
        head = s[:gt + 1]
        out = outl.append
        ts = self.tool_state
        if _PARAM_END.fullmatch(head):
            if ts == "param":
                self.buf, self.tool_state = s[gt + 1:], "invoke"
                out("\n</parameter>\n")
                return True
            return None
        if ts == "param":
            return None                         # inside a value only its closing tag means anything
        if _CALLS.fullmatch(head):
            self.buf, self.tool_state = s[gt + 1:], "calls"
            return True
        m = _INVOKE.fullmatch(head)
        if m:
            if ts == "invoke":
                out("</function>\n</tool_call>\n")
            name = m.group("name") or self.tool
            if getattr(self, "sep", False):
                out("\n\n")
                self.sep = False
            self.buf, self.tool_state, self.tool = s[gt + 1:], "invoke", name
            if self.on_open is not None:
                self.on_open(name)
            out(f"<tool_call>\n<function={name}>\n")
            return True
        if _PARAM.fullmatch(head) and ts == "invoke":
            self.buf, self.tool_state = s[gt + 1:], "param"
            out(f"<parameter={_PARAM.fullmatch(head).group('name')}>\n")
            return True
        if _INVOKE_END.fullmatch(head):
            if ts == "invoke":
                self.buf, self.tool_state = s[gt + 1:], "calls"
                out("</function>\n</tool_call>\n")
                return True
            return None
        if _CALLS_END.fullmatch(head):
            if ts == "invoke":
                out("</function>\n</tool_call>\n")
            if ts in ("calls", "invoke"):
                self.buf, self.tool_state = s[gt + 1:], "out"
                return True
            return None
        return None


class AtemOutputParser(OutputParser):
    """OutputParser for Muse Glimmer: its channels and ATEM calls are read as the template-form text they are."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opened: set[str] = set()
        self.atem = AtemTranslator(thinking=self.state == "reasoning", on_open=self.opened.add)
        self._reasoning: list[str] = []
        self._answered = False

    def _watch(self, evs):
        for ev in evs:
            if ev.kind == "reasoning":
                self._reasoning.append(ev.text)
            elif ev.kind in ("tool_start", "tool_call") or (ev.kind == "content" and ev.text.strip()):
                self._answered = True
        return evs

    def _callable(self, name: str) -> bool:
        """A call the model wrote as an ATEM invoke is a call, declared or not (the client can answer it with an error;
        as text it would read as the answer) - as serve/dsml.py does for DSML calls."""
        return super()._callable(name) or name in self.opened

    def feed(self, delta: str):
        return self._watch(super().feed(self.atem.feed(delta)))

    def finish(self, reason: str | None = None):
        if reason not in (None, "stop") and self.atem.held:
            self.atem.buf = ""              # cut inside a marker or tag: no text the model meant
        rest = self.atem.flush()
        out = self._watch(super().feed(rest) if rest else [])
        out += self._watch(super().finish(reason))
        text = "".join(self._reasoning).strip()
        if reason in (None, "stop") and text and not self._answered:
            # the turn ended with thinking only (no to=user message): the thinking is the answer
            self._answered = True
            out.append(Event("content", text))
        return out

    @property
    def clean(self) -> bool:
        return not self.buf and not self.atem.held


def to_atem_force(force: str | None) -> str | None:
    """forced_call's opening of a call in the Qwen form -> the same opening as a Glimmer tool message (it follows the
    prompt's `<|start|>assistant`, or the wrap-up's)."""
    if not force:
        return force
    at = force.find(FUNC_START)
    name = force[at + len(FUNC_START):] if at >= 0 else ""
    if name.endswith(">\n"):
        name = name[:-2]
        return f' to={name}{MESSAGE}<atem:function_calls>\n<atem:invoke name="{name}">\n'
    return f" to={name}"


__all__ = ["AtemTranslator", "AtemOutputParser", "to_atem_force", "REASONING_CLOSE", "REASONING_WRAP_UP", "ANSWER_OPEN",
           "CALL_START"]
