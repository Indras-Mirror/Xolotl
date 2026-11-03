"""serve/dsml.py - DeepSeek-V4.1's tool-call text (DSML) for the server.

DeepSeek-V4.1 writes a tool call as (the model's own chat template, the same bytes as DeepSeek's encoding.py)

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="get_weather">
    <｜DSML｜ parameter name="city" string="true">Zagreb</｜DSML｜ parameter>
    <｜DSML｜ parameter name="days" string="false">3</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

`｜DSML｜` is one token (id 128825, a user-defined one, so it is also text when a request is tokenized).  String
values are written as they are; every other type is JSON (string="false").

The server's OutputParser (serve/frontend.py) reads the Qwen form `<tool_call><function=NAME><parameter=P>...`, with
its streaming of tool arguments, its calls inside the thinking and its recovery rules.  DsmlTranslator rewrites the
DSML text into that form as it streams (holding back only a tag that is still arriving), and DsmlOutputParser is the
OutputParser behind it.  So the OpenAI and Anthropic paths, tool_choice, streaming arguments and the thinking/answer
split behave for DeepSeek as they do for the other families, and there is one place that knows the DSML spelling.
"""
from __future__ import annotations

import re
from pathlib import Path

from serve.frontend import CALL_START, FUNC_START, Event, OutputParser, _declared, _tool_properties

DSML = "｜DSML｜"
_OPEN = "<" + DSML
_CLOSE = "</" + DSML
_TAG_MAX = 600                  # a tag (a tool name, a parameter name) longer than this is text, not held any further

# The tag marker.  The model's own is the ｜DSML｜ token; low-bit quants also write the tags without it - `< invoke
# name="write">` - or with a backslash (ASCII or U+FF3C) where the token goes (2026-11-02, UD-IQ1_M: such calls reached
# clients as text).  group "m": the token (any spacing after it), a backslash (any spacing), or nothing - then at least
# one space, so `<invoke>` / `<calls>` in prose are never tags.  A tag without the token opens a call at the start of a line
# outside a code fence, or elsewhere for a tool the request declared (DsmlTranslator's `known`).
_M = r"(?P<m>｜DSML｜\s*|[\\＼｜]DSML[\\＼｜]\s*|[\\＼]\s*|\s+)"   # ... or the token misspelled: ＼DSML＼, \DSML\
_CALLS = re.compile(rf"<{_M}(?:tool_|function_)?calls\s*>")
_CALLS_END = re.compile(rf"</{_M}(?:tool_|function_)?calls\s*>")
_INVOKE = re.compile(rf'<{_M}invoke\s+name\s*=\s*"(?P<name>[^"]*)"\s*>')
_INVOKE_END = re.compile(rf"</{_M}invoke\s*>")
_PARAM = re.compile(rf'<{_M}parameter\s+name\s*=\s*"(?P<name>[^"]*)"(?:\s+string\s*=\s*"(?P<s>true|false)")?\s*>')
_PARAM_END = re.compile(rf"</{_M}parameter\s*>")
# a tag cut short is held while it can still become one of those: "<", "<｜DS", '<｜DSML｜ invoke name="get_we'
_PARTIAL = re.compile(r"^</?(?:｜(?:D(?:S(?:M(?:L(?:｜[^<>]*)?)?)?)?)?)?$")
# ... and in the token-less forms: "< ", "<\\", "< inv", '< invoke name="wr', "</ param"
_BARE_HEAD = re.compile(r"^</?(?:[\\＼]\s*|\s+)(?P<rest>[^<>]*)$")
_ALT_PARTIAL = re.compile(r"^</?[\\＼｜](?:D(?:S(?:M(?:L)?)?)?)?$")    # "<＼", "<＼DSM"
_ALT = re.compile(r"^(</?)[\\＼｜]DSML[\\＼｜]")
_WORDS = ("calls", "tool_calls", "function_calls", "invoke", "parameter")


def _could_still_be_a_tag(s: str) -> bool:
    """`s` starts with '<' and has no '>' yet: it may be the start of a DSML tag."""
    if len(s) > _TAG_MAX:
        return False
    if _PARTIAL.match(s) is not None:
        return True
    if _ALT_PARTIAL.match(s) is not None:
        return True
    s = _ALT.sub(r"\1 ", s, count=1)          # `<＼DSML＼ inv` is held as `< inv` is
    m = _BARE_HEAD.match(s)
    if m is None:
        return s in ("<\\", "</\\", "<＼", "</＼")
    rest = m.group("rest")
    if any(w.startswith(rest) for w in _WORDS):
        return True
    for w in ("invoke", "parameter"):          # the word is whole: its attributes are still arriving
        if rest.startswith(w) and (rest == w or rest[len(w)].isspace()) and "\n" not in rest:
            return True
    return any(rest.startswith(w) and rest[len(w):].strip() == "" for w in _WORDS[:3])


def _is_tag_start(head: str) -> bool:
    """`head` (up to its '>') starts like a DSML tag: `<｜DSML｜`, `<\\`, `<＼` or `< ` (closing forms too)."""
    i = 2 if head.startswith("</") else 1
    return head.startswith(DSML, i) or head[i:i + 1] in ("\\", "＼", "｜") or head[i:i + 1].isspace()


def _qwen_opener_cut(held: str) -> bool:
    """The parser's held text is the start of a call in the Qwen form and nothing else: `<tool_`, `<tool_call>\n<func`,
    `<function=get_w` (what the output ends on when max_tokens lands in a call's opener)."""
    s = held.lstrip()
    if not s:
        return False
    if any(t.startswith(s) for t in (CALL_START, FUNC_START)):
        return True
    if s.startswith(CALL_START):
        s = s[len(CALL_START):].lstrip()
        if not s or FUNC_START.startswith(s):
            return True
    return s.startswith(FUNC_START) and ">" not in s and "\n" not in s


class DsmlTranslator:
    """DSML text -> the Qwen tool-call text, a delta at a time.  Text outside a call passes through unchanged.

    on_param(tool, name, is_string) is called when a parameter tag is read, before its text is returned, so the parser
    behind it can treat a string="true" value as the string it is."""

    def __init__(self, on_param=None, known=None):
        self.buf = ""
        self.state = "text"        # text | calls (inside the block) | invoke | param (inside a value)
        self.tool = None
        self.on_param = on_param
        self.known = known         # tool name -> declared?  A call opened without the DSML token needs a declared tool
        self.opened: set[str] = set()    # tools a call was opened for with the token, in a calls block or alone on a line: calls, declared or not
        self.line = ""             # the text of the current line, and whether it is inside a code fence
        self.fenced = False

    @property
    def held(self) -> str:
        """Text held back because it may be the start of a tag."""
        return self.buf

    def feed(self, delta: str) -> str:
        self.buf += delta
        return self._run(final=False)

    def flush(self) -> str:
        """The end of the output: whatever is held is text."""
        return self._run(final=True)

    # ------------------------------------------------------------------------------------------------
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
            tag = self._tag(out)
            if tag == "wait":
                if final:
                    self._text(self.buf, out)
                    self.buf = ""
                break
            if tag is None:                     # an ordinary '<'
                self._text("<", out)
                self.buf = self.buf[1:]
        return "".join(out)

    def _text(self, text: str, out: list[str]) -> None:
        if self.state in ("calls", "invoke"):
            if text.strip() == "":
                return                          # the layout between tags
            # stray text inside the block: it is shown as text, and the block goes on
        out.append(text)
        *done, self.line = (self.line + text).split("\n")
        for line in done:
            if line.lstrip().startswith("```"):
                self.fenced = not self.fenced

    def _tag(self, outl: list[str]):
        """Handle the tag at the start of self.buf.  Returns True (a tag was consumed), "wait" (it may still become
        one) or None (it is not one)."""
        s = self.buf
        gt = s.find(">")
        if gt < 0:
            return "wait" if _could_still_be_a_tag(s) else None
        head = s[:gt + 1]
        if not _is_tag_start(head):
            return None
        out = outl.append
        m = _PARAM_END.fullmatch(head)
        if m:
            if self.state == "param":
                self.buf = s[gt + 1:]
                self.state = "invoke"
                out("\n</parameter>\n")
                return True
            return None
        m = _INVOKE_END.fullmatch(head) or _CALLS_END.fullmatch(head)
        if m:
            closing_invoke = bool(_INVOKE_END.fullmatch(head))
            if self.state == "param":           # the value's closing tag was left out
                out("\n</parameter>\n")
                self.state = "invoke"
            if closing_invoke and self.state == "invoke":
                self.buf = s[gt + 1:]
                self.state = "calls"
                out("</function>\n</tool_call>\n")
                return True
            if not closing_invoke and self.state in ("calls", "invoke"):
                if self.state == "invoke":
                    out("</function>\n</tool_call>\n")
                self.buf = s[gt + 1:]
                self.state = "text"
                return True
            return None
        if self.state == "param":
            return None                         # inside a value only a closing tag means anything
        m = _CALLS.fullmatch(head)
        if m:
            self.buf = s[gt + 1:]
            self.state = "calls"
            return True
        m = _INVOKE.fullmatch(head)
        if m:
            name = m.group("name")
            # the token, inside a calls block, or at the start of a line outside a code fence: in the middle of a line or
            # in a fence, `< invoke name="X">` is what the model is talking about, a call only for a declared tool
            sure = (m.group("m").startswith(DSML) or self.state in ("calls", "invoke")
                    or (not self.line.strip() and not self.fenced))
            if not sure and not (self.known is not None and self.known(name)):
                return None
            if sure:
                self.opened.add(name)
            if self.state == "invoke":          # the previous call's closing tag was left out
                out("</function>\n</tool_call>\n")
            self.buf = s[gt + 1:]
            self.state, self.tool = "invoke", name
            out(f"<tool_call>\n<function={name}>\n")
            return True
        m = _PARAM.fullmatch(head)
        if m and self.state in ("calls", "invoke"):
            self.buf = s[gt + 1:]
            self.state = "param"
            if self.on_param is not None:
                self.on_param(self.tool, m.group("name"), m.group("s") != "false")
            out(f"<parameter={m.group('name')}>\n")
            return True
        return None


class DsmlOutputParser(OutputParser):
    """OutputParser for DeepSeek-V4.1: the model's DSML calls are read as the template-form calls they are."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dsml = DsmlTranslator(self._on_param, known=lambda name: name in self.schemas)
        # The 2-bit sets often write the whole answer after <think> and stop (EOS) without </think>: everything would be reasoning and
        # the reply empty (P2003 install test, 4 of 4 prompts).  The reasoning text is kept here; finish() hands it over as the answer
        # when the turn ended on a stop with no answer and no call.
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
        """A call the model opened with the DSML token (or inside a calls block) is a call even for a tool the request did
        not declare.  The token is never text, so the base rule (#3075: an undeclared name is text) sent it to the content as `<tool_call>` markup
        and the client read it as the answer (2026-11-03: history with a `bash` call, tools = [write] only: the model
        called bash again, 4 of 6 replies were that markup).  The client gets the call it can answer with an error.
        A call the model wrote in the Qwen form is one too where it acts: at the start of a line, outside code."""
        if super()._callable(name) or name in self.dsml.opened:
            return True
        at = self.buf.find(CALL_START)
        return self.state == "content" and at >= 0 and self._ok_at(at)

    def _bare_opener(self) -> tuple[int, bool]:
        """... and so is `<function=NAME>` with no wrapper, where the base parser wants a declared tool.  Not on the line
        after a `<tool_call>` that was already read as text: that is the rest of a quote."""
        found = super()._bare_opener()
        if found[0] >= 0:
            return found
        p = self.buf.find(FUNC_START)
        while p >= 0:
            quoted = (self.line + self.buf[:p]).rstrip().endswith(CALL_START)
            if not quoted and self.buf.find(">", p) >= 0 and self._ok_at(p):
                return p, True
            p = self.buf.find(FUNC_START, p + 1)
        return -1, False

    def _on_param(self, tool, name, is_string) -> None:
        """string="true": the value is a string whatever it looks like ("123", "true"), as the template wrote it."""
        if not is_string:
            return
        spec = self.schemas.get(tool)
        if spec is None:
            return
        params = dict(spec.get("parameters") or {})
        props = dict(_tool_properties(spec))          # a union's variants too (#208)
        if _declared(props.get(name)) != "string":
            props[name] = {**(props.get(name) or {}), "type": "string"}
            params["properties"] = props
            self.schemas[tool] = {**spec, "parameters": params}
        if self.scall is not None and self.scall.name == tool:
            self.sdeclared[name] = "string"

    def feed(self, delta: str):
        return self._watch(super().feed(self.dsml.feed(delta)))

    def finish(self, reason: str | None = None):
        if reason not in (None, "stop") and self.dsml.held:
            # The turn was cut (max_tokens, a cancel) inside a DSML tag: `<｜DSML｜ invoke nam` before the call's name was
            # complete, or `</｜DSML｜ ` after a whole call.  It is no text the model meant, and as text it reached clients
            # as DSML in the content (an agent read it as the answer).  Held text is only ever a tag that may still be
            # arriving (_could_still_be_a_tag), so this drops nothing else; a natural stop keeps it as text, as before.
            self.dsml.buf = ""
        rest = self.dsml.flush()
        out = self._watch(super().feed(rest) if rest else [])
        if reason not in (None, "stop") and self.state == "content" and _qwen_opener_cut(self.buf):
            self.buf = ""                  # ... and inside the opener of a call the model wrote in the Qwen form (`<tool_`)
        elif reason not in (None, "stop") and self.state in ("call", "json") and self.scall is None:
            self.buf = ""                  # ... or after it, before the call had a name (`<tool_call>\n<func`): no call to report
        elif reason not in (None, "stop") and self.state == "call" and self.stream_tools and self.scall is not None:
            # ... or a call already announced: it stays the unfinished call the stream shows (tool_start + its arguments,
            # no tool_call).  The base parser sends a call cut right after its </function> as text, markup and all.
            out += self._watch(self._scan())
            self.buf, self.state = "", "content"
            self._reset_scan()
        out += self._watch(super().finish(reason))
        text = "".join(self._reasoning).strip()
        if reason in (None, "stop") and text and not self._answered:
            # no </think> and no answer: the text IS the answer.  A streaming client has already seen it as reasoning and now gets
            # it as content; the non-streaming collectors (server.py) drop the identical reasoning.
            self._answered = True
            out.append(Event("content", text))
        return out

    @property
    def clean(self) -> bool:
        """No tag is held back anywhere: a clean point to stop the thinking at."""
        return not self.buf and not self.dsml.held


FIXTURE_TEMPLATE = Path(__file__).resolve().parent / "fixtures" / "deepseek41" / "chat_template.jinja"


def v41_template_ok(text: str) -> bool:
    """The V4.1 model writes its call block as `<｜DSML｜ calls>` / `<｜DSML｜ invoke ...` (a space after the DSML token).
    Packs made from another GGUF ship the V4-Flash template (`<｜DSML｜tool_calls>`, no space): the model then sees one
    spelling in the prompt and writes another, and only 2 of 11 tool-call tries came back as real calls (P2003 install
    test; 0 of 4 vs 6 of 6 with this template on the same pack)."""
    return " calls>" in text


def pick_template(pack_template: Path | None) -> tuple[Path, bool]:
    """(the template file to render with, True when it replaced a pack template that spells the call block otherwise)."""
    if pack_template is not None and pack_template.exists():
        try:
            if v41_template_ok(pack_template.read_text(encoding="utf-8")):
                return pack_template, False
        except OSError:
            return pack_template, False
        return FIXTURE_TEMPLATE, True
    return FIXTURE_TEMPLATE, False


def to_dsml_force(force: str | None) -> str | None:
    """forced_call's opening of a call in the Qwen form -> the same opening in DSML (the model's own spelling)."""
    if not force:
        return force
    head = _OPEN + " calls>\n"
    marker = "<function="
    at = force.find(marker)
    name = force[at + len(marker):] if at >= 0 else ""
    if name.endswith(">\n"):
        return head + f'{_OPEN} invoke name="{name[:-2]}">\n'
    return head + f'{_OPEN} invoke name="{name}'
