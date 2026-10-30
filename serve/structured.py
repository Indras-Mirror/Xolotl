"""Structured request normalization, GBNF conversion and final JSON validation.

Engines advertising INFO grammar=1 constrain decoding; older engines retain the
prompt-and-validate response_format path. Failed JSON is never delivered as success.

`jsonschema` is optional (no new hard dependency of the server): json_object needs only the standard library, and
json_schema is checked against its schema when the package is installed - without it, only that the answer is one
JSON object, and the server says so once.
"""
import json
import copy
import threading


class StructuredOutputError(RuntimeError):
    pass


def normalize_request(req):
    """Return (GBNF, response_format), rejecting ambiguous or unsupported constraints.

    The caller chooses whether to compile the schema after discovering INFO.
    Only this function handles public aliases; protocol paths are server-owned.
    """
    specs = []
    fmt = req.get("response_format")
    if fmt is not None:
        if not isinstance(fmt, dict):
            raise ValueError("response_format must be an object")
        if fmt.get("type") != "text":
            specs.append(("format", fmt))
    structured = req.get("structured_outputs")
    if structured is not None:
        if not isinstance(structured, dict) or not structured:
            raise ValueError("structured_outputs must be a non-empty object")
        for key, value in structured.items():
            if key == "disable_any_whitespace":
                if not isinstance(value, bool):
                    raise ValueError("structured_outputs.disable_any_whitespace must be boolean")
                continue
            if key not in ("json", "grammar", "choice", "regex", "json_object"):
                raise ValueError(f"structured_outputs.{key} is not supported")
            specs.append((key, value))
    for key, kind in (("guided_json", "json"), ("guided_grammar", "grammar"),
                      ("guided_choice", "choice"), ("guided_regex", "regex"),
                      ("grammar", "grammar"), ("json_schema", "json")):
        if key in req and req[key] is not None:
            specs.append((kind, req[key]))
    if not specs:
        if structured is not None:
            raise ValueError("structured_outputs needs an output constraint")
        return None, fmt
    # Some clients send the same GBNF in llama.cpp and vLLM fields together.
    if len(specs) > 1 and all(kind == "grammar" for kind, _ in specs):
        if any(value != specs[0][1] for _, value in specs[1:]):
            raise ValueError("conflicting grammar texts in structured output fields")
        specs = specs[:1]
    if len(specs) != 1:
        raise ValueError("specify only one structured output constraint")
    kind, value = specs[0]
    if kind == "format":
        return None, value
    if kind == "regex":
        raise ValueError("structured output regex is not supported; use GBNF grammar")
    if kind == "json_object":
        if value is not True:
            raise ValueError("structured_outputs.json_object must be true")
        return None, {"type": "json_object"}
    if kind == "json":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ValueError("structured JSON schema must be an object or JSON schema string") from exc
        if not isinstance(value, dict):
            raise ValueError("structured JSON schema must be an object")
        return None, {"type": "json_schema", "json_schema": {"name": "structured_output", "schema": value}}
    if kind == "choice":
        if not isinstance(value, list) or not value or not all(isinstance(s, str) and s for s in value):
            raise ValueError("structured choice must be a non-empty list of non-empty strings")
        value = "root ::= " + " | ".join(json.dumps(s, ensure_ascii=False) for s in value)
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError("grammar must be non-empty GBNF without NUL bytes")
    return value, None


def format_grammar(fmt, compact=False):
    """llama.cpp's MIT converter; no remote fetches, no mutation of caller schemas."""
    from serve.json_schema_to_grammar import SchemaConverter
    schema = {"type": "object", "additionalProperties": True} if fmt["type"] == "json_object" else fmt["json_schema"]["schema"]
    converter = SchemaConverter(prop_order={}, allow_fetch=False, dotall=False, raw_pattern=False)
    try:
        schema = converter.resolve_refs(copy.deepcopy(schema), "request")
        converter.visit(schema, "")
        if compact:
            converter._rules["space"] = '""'
        return converter.format_grammar()
    except (AssertionError, KeyError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"unsupported JSON schema grammar: {exc}") from exc


class _ObjectOnly:
    """The validator for json_object, and for json_schema without jsonschema: one JSON object."""

    def iter_errors(self, value):
        if not isinstance(value, dict):
            yield _ObjectError()


class _ObjectError:
    absolute_path = ()
    message = "the answer is not a JSON object"


_jsonschema = None                     # (validators, SchemaError, Registry, NoSuchResource), False when not installed
_jsonschema_lock = threading.Lock()


def jsonschema_modules():
    """jsonschema, imported on first use; None when it is not installed (said once, in the server window)."""
    global _jsonschema
    with _jsonschema_lock:
        if _jsonschema is None:
            try:
                from jsonschema import validators
                from jsonschema.exceptions import SchemaError
                from referencing import Registry
                from referencing.exceptions import NoSuchResource
                _jsonschema = (validators, SchemaError, Registry, NoSuchResource)
            except ImportError:
                _jsonschema = False
                print('[strata] response_format json_schema: the Python package jsonschema is not installed, so '
                      'answers are only checked to be one JSON object (python -m pip install "jsonschema>=4.23,<5")',
                      flush=True)
        return _jsonschema or None


def _only_objects(node, root, refs=()):
    """True when every value `node` accepts is a JSON object, so "return one JSON object" stays true.

    `type: object`, an anyOf/oneOf whose branches all qualify (e.g. a root union of object shapes, which llama.cpp's
    grammar path accepts and apps send), an allOf with a qualifying member, or a local `$ref` to one of these.
    Anything that can also be an array, string, number, boolean or null (including `type: ["object", "null"]`) is
    not, and a `$ref` cycle never qualifies.
    """
    if not isinstance(node, dict):
        return False
    if node.get("type") == "object":
        return True
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#") and ref not in refs:
        target = root
        for part in ref[1:].split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return False
            target = target[part]
        return _only_objects(target, root, refs + (ref,))
    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if isinstance(branches, list) and branches and all(_only_objects(b, root, refs) for b in branches):
            return True
    branches = node.get("allOf")
    return isinstance(branches, list) and any(_only_objects(b, root, refs) for b in branches)


def prepare_format(response_format, messages, with_tools=False):
    if response_format is None:
        return messages, None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "text":
        return messages, None
    if kind == "json_object":
        schema = {"type": "object"}
        modules = None                 # the standard library is enough for "one JSON object"
    elif kind == "json_schema":
        spec = response_format.get("json_schema")
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise ValueError("response_format.json_schema needs a schema object")
        if not isinstance(spec.get("name"), str) or not spec["name"]:
            raise ValueError("response_format.json_schema needs a name")
        if "strict" in spec and not isinstance(spec["strict"], bool):
            raise ValueError("response_format.json_schema.strict must be boolean")
        schema = spec["schema"]
        if not _only_objects(schema, schema):
            raise ValueError("response_format schema must accept only JSON objects at its root "
                             "(type object, or anyOf/oneOf of object schemas)")
        modules = jsonschema_modules()
    else:
        raise ValueError("response_format.type must be text, json_object or json_schema")

    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef") and isinstance(value, str) and not value.startswith("#"):
                    raise ValueError("response_format supports only local schema references (#...)")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)
    check_refs(schema)
    if modules is None:
        validator = _ObjectOnly()
    else:
        validators, SchemaError, Registry, NoSuchResource = modules

        def no_remote(uri):
            raise NoSuchResource(ref=uri)

        try:
            cls = validators.validator_for(schema)
            cls.check_schema(schema)
            validator = cls(schema, registry=Registry(retrieve=no_remote))
        except SchemaError as exc:
            raise ValueError(f"invalid response_format schema: {exc.message}") from exc
    directive = ("OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
                 "No Markdown, headings, code fences, commentary, or text outside the JSON. "
                 "Use every required field, correct types, and only allowed fields. "
                 "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n" +
                 json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    if with_tools:
        # /v1/responses (#782): the schema is for the final answer; a turn that calls a tool is not an answer
        directive += ("\nThis applies only to your final answer. To use a tool, call it as usual; the JSON object is "
                      "what you write once you are done with the tools.")
    messages = [dict(message) for message in messages]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        if isinstance(content, list):
            messages[0]["content"] = content + [{"type": "text", "text": directive}]
        else:
            messages[0]["content"] = content + "\n\n" + directive
    else:
        messages.insert(0, {"role": "system", "content": directive})
    return messages, validator


def _extract_json(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        first = s.find("\n")
        if first != -1:
            end = s.find("```", first + 1)
            if end != -1:
                s = s[first + 1:end].strip()
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return s[start:]


def validated_json(text, validator, finish):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    if finish != "stop":
        raise StructuredOutputError(f"structured output was incomplete (finish_reason={finish}); increase the output budget")
    try:
        value = json.loads(_extract_json(text), object_pairs_hook=pairs, parse_constant=constant)
        canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError(f"model did not return valid JSON: {exc}") from exc
    try:
        error = next(validator.iter_errors(value), None)
    except Exception as exc:
        raise StructuredOutputError(f"could not validate structured output: {exc}") from exc
    if error is not None:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise StructuredOutputError(f"model output failed the JSON Schema at {path}: {error.message}")
    return canonical
