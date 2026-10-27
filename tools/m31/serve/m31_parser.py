"""Incremental MiniMax reasoning and recursive XML tool-call parsing.

MiniMax's namespace is a literal model token, not an XML namespace. String
arguments remain literal (including ampersands and angle brackets); objects and
arrays follow the supplied JSON schema, matching the GGUF template's to_xml.
"""
import json

from serve.frontend import Event, ToolCall

NS = "]<]minimax[>["
THINK_START, THINK_END = "<mm:think>", "</mm:think>"
CALL_START, CALL_END = NS + "<tool_call>", NS + "</tool_call>"
INVOKE_END = NS + "</invoke>"
CONTROL_PIECES = (NS, THINK_START, THINK_END, "<tool_call>", "</tool_call>")


class TokenDelta(str):
    """Decoded text with positions of actual special IDs, before text loses token identity."""
    def __new__(cls, text, controls=None):
        value = super().__new__(cls, text)
        value.controls = dict(controls or {})
        return value


def structural(text, at, literal, controls):
    if controls is None:
        return True
    for piece in CONTROL_PIECES:
        offset = literal.find(piece)
        if offset >= 0 and controls.get(at + offset) != piece:
            return False
    return True


class Incomplete(ValueError):
    pass


class _Reader:
    def __init__(self, text, schema, controls=None):
        self.text, self.at, self.root, self.controls = text, 0, schema, controls

    def space(self):
        while self.at < len(self.text) and self.text[self.at].isspace():
            self.at += 1

    def literal(self, literal):
        if self.text.startswith(literal, self.at) and structural(self.text, self.at, literal, self.controls):
            self.at += len(literal)
        elif literal.startswith(self.text[self.at:]):
            raise Incomplete("incomplete MiniMax tool tag")
        else:
            raise ValueError("malformed MiniMax tool tag at character %d" % self.at)

    def at_close(self, close):
        if self.text.startswith(close, self.at) and structural(self.text, self.at, close, self.controls):
            return True
        if close.startswith(self.text[self.at:]):
            raise Incomplete("incomplete MiniMax closing tag")
        return False

    def schema(self, schema):
        seen = set()
        while "$ref" in schema:
            ref = schema["$ref"]
            if ref in seen or not ref.startswith("#/"):
                raise ValueError("unsupported or recursive MiniMax tool schema reference")
            seen.add(ref)
            schema = self.root
            for part in ref[2:].split("/"):
                schema = schema[part.replace("~1", "/").replace("~0", "~")]
        return schema

    def value(self, schema, close):
        schema = self.schema(schema)
        kind = schema.get("type")
        alternatives = schema.get("anyOf", schema.get("oneOf", []))
        if kind == "string" or isinstance(kind, list) and "string" in kind or any(
                self.schema(alt).get("type") == "string" for alt in alternatives):
            return self.string(close)
        if alternatives:
            errors, start = [], self.at
            for alternative in alternatives:
                self.at = start
                try:
                    return self.value(alternative, close)
                except ValueError as exc:
                    errors.append(exc)
            self.at = start
            raise next((exc for exc in errors if isinstance(exc, Incomplete)), errors[0])
        if kind == "object" and schema.get("properties"):
            value = self.members(schema, close)
        elif kind == "array":
            value = []
            self.space()
            while not self.at_close(close):
                self.literal(NS + "<item>")
                value.append(self.value(schema.get("items", {}), NS + "</item>"))
                self.space()
        else:
            raw = self.string(close)
            try:
                return json.loads(raw)
            except ValueError as exc:
                raise ValueError("invalid JSON scalar in MiniMax tool argument") from exc
        self.space()
        self.literal(close)
        return value

    def string(self, close):
        end = self.text.find(close, self.at)
        while end >= 0 and not structural(self.text, end, close, self.controls):
            end = self.text.find(close, end + 1)
        if end < 0:
            raise Incomplete("incomplete MiniMax tool argument")
        value = self.text[self.at:end]
        self.at = end + len(close)
        return value

    def members(self, schema, close):
        value, properties = {}, schema.get("properties", {})
        self.space()
        while not self.at_close(close):
            self.literal(NS + "<")
            end = self.text.find(">", self.at)
            if end < 0:
                raise Incomplete("incomplete MiniMax parameter name")
            name = self.text[self.at:end]
            self.at = end + 1
            if name not in properties or name in value:
                raise ValueError("unknown or repeated MiniMax parameter %r" % name)
            value[name] = self.value(properties[name], NS + "</" + name + ">")
            self.space()
        missing = set(schema.get("required", [])) - value.keys()
        if missing:
            raise ValueError("missing MiniMax parameters: " + ", ".join(sorted(missing)))
        return value


def parse_calls(text, schemas, controls=None):
    reader = _Reader(text, {}, controls)
    reader.literal(CALL_START)
    reader.space()
    calls = []
    while not reader.at_close(CALL_END):
        reader.literal(NS + '<invoke name="')
        end = text.find('\">', reader.at)
        if end < 0:
            raise Incomplete("incomplete MiniMax invocation")
        name = text[reader.at:end]
        reader.at = end + 2
        if name not in schemas:
            raise ValueError("unknown MiniMax tool %r" % name)
        schema = schemas[name].get("parameters", {})
        reader.root = schema
        arguments = reader.members(reader.schema(schema), INVOKE_END)
        reader.literal(INVOKE_END)
        reader.space()
        calls.append(ToolCall(name=name, arguments=arguments))
    reader.literal(CALL_END)
    if not calls:
        raise ValueError("empty MiniMax tool-call block")
    return calls, reader.at


class M31OutputParser:
    def __init__(self, thinking=True, tools=None, stream_tools=False, recover=False):
        if recover:
            raise ValueError("Qwen tool-call recovery is unavailable for MiniMax")
        self.schemas = {tool.get("name", (tool.get("function") or {}).get("name")):
                        tool.get("function", tool) for tool in tools or []}
        self.thinking = thinking
        self.state, self.buf, self.pending = "initial", "", []
        self.rescued = self.refused = 0
        self._reasoning_space = False
        self._controls = {}

    def _append(self, delta):
        offset = len(self.buf)
        self.buf += str(delta)
        if isinstance(delta, TokenDelta):
            self._controls.update({offset + at: piece for at, piece in delta.controls.items()})
        else:
            # Text-only oracle fixtures and prefixes intentionally injected by
            # Service.run have no token IDs. Model output always uses TokenDelta.
            for piece in CONTROL_PIECES:
                at = self.buf.find(piece)
                while at >= 0:
                    self._controls[at] = piece
                    at = self.buf.find(piece, at + 1)

    def _take(self, size):
        text, self.buf = self.buf[:size], self.buf[size:]
        self._controls = {at - size: piece for at, piece in self._controls.items() if at >= size}
        return text

    def _find(self, literal):
        at = self.buf.find(literal)
        while at >= 0 and not structural(self.buf, at, literal, self._controls):
            at = self.buf.find(literal, at + 1)
        return at

    @staticmethod
    def _hold(text, markers):
        return max((size for marker in markers for size in range(1, min(len(marker), len(text)) + 1)
                    if text.endswith(marker[:size])), default=0)

    def feed(self, delta):
        self._append(delta)
        out = []
        while self.buf:
            if self.state == "initial":
                if self._find(THINK_START) == 0:
                    self._take(len(THINK_START))
                    self.state, self._reasoning_space = "reasoning", True
                elif self._find(THINK_END) == 0:
                    self._take(len(THINK_END))
                    self.state = "content"
                elif any(len(self.buf) < len(marker) and marker.startswith(self.buf) for marker in (THINK_START, THINK_END)):
                    break
                else:
                    self.state = "content"
            elif self.state == "reasoning":
                if self._reasoning_space:
                    self._take(len(self.buf) - len(self.buf.lstrip()))
                    self._reasoning_space = not bool(self.buf)
                end = self._find(THINK_END)
                size = end if end >= 0 else len(self.buf) - self._hold(self.buf, (THINK_END,))
                text = self._take(size)
                if text and self.thinking:
                    out.append(Event("reasoning", text))
                if end < 0:
                    break
                self._take(len(THINK_END))
                self.state = "content"
            else:
                if self.schemas and self._find(CALL_START) == 0:
                    try:
                        calls, size = parse_calls(self.buf, self.schemas, self._controls)
                    except Incomplete:
                        break
                    out.extend(Event("tool_call", call=call) for call in calls)
                    self._take(size)
                    continue
                end = self._find(CALL_START) if self.schemas else -1
                size = end if end >= 0 else len(self.buf) - self._hold(self.buf, (CALL_START,) if self.schemas else ())
                if size:
                    out.append(Event("content", self._take(size)))
                if end < 0:
                    break
        return out

    def finish(self, reason=None):
        if self.schemas and self._find(CALL_START) == 0:
            raise ValueError("incomplete MiniMax tool call at end of generation")
        text = self._take(len(self.buf))
        if not text or self.state == "reasoning" and not self.thinking:
            return []
        return [Event("reasoning" if self.state == "reasoning" else "content", text)]


def forced_call(tool_choice, tools):
    from serve.frontend import tool_choice_of
    kind, name = tool_choice_of(tool_choice)
    names = {tool.get("name", (tool.get("function") or {}).get("name")) for tool in tools or []}
    if kind in ("auto", "none"):
        return None
    if kind == "required" and len(names) == 1:
        name = next(iter(names))
    if name in names:
        return CALL_START + "\n" + NS + '<invoke name="' + name + '\">'
    return CALL_START + "\n" + NS + '<invoke name="' if kind == "required" and names else None
