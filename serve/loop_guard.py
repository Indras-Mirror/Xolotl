"""serve/loop_guard.py - the tool-call loop guard: an agent that makes the same tool call - the same name
and the same arguments - again and again, turn after turn, gets one short note before its next turn: it is repeating
itself, so it should try something else or stop.

The design follows Project Maya's serve/loop_guard.py (MIT; its `tool_repeat` and `tool_note`): the conversation the
client sends is read for the repeated call, the note is a message of its own after the last tool result (inside the
result a model ran the call again; as a message of its own it moved on), and only this request's prompt changes - the
client's conversation is not touched, so its next request matches the cached prompt up to that result.  Simpler than
Maya's: no blocking of calls, no results compared - a run of the same call N times in a row is enough.

    repeated(messages, n)   -> (tool names, times) when the request answers the results of a call that is the same as
                               the ones before it, N times in a row (else None)
    note(names, times)      the note's text
    setting(value)          the run config's "loop_guard": absent or true -> REPEATS, {"repeats": N} -> N, false -> off

Calls are compared by name and arguments in one form (args_key: a client's JSON string and a parsed dict compare
equal; key order does not matter).  An assistant turn with several calls counts as one step: the same batch again is a
repeat.  An assistant turn without a call ends the run (the model said something in between), and so does a different
call.  DeepSeek writes its calls as DSML (#844 in Maya); the client sends them back as JSON like any other, so nothing
here depends on the template.

0.95, the window (after Maya 770ddef #844 and 989ee69 #717, MIT): a run in a row misses a loop of two calls in turn
(read A, read B, read A, read B ...) and one with another call in between.  So the calls since the user's last
message are also counted over a window: a call of the last turn that the last WINDOW calls hold TIMES times, with
the same arguments and the same result, is a loop too (windowed / window_note).  A result counts as the same when it
is the same text - or, for the same call made again right after itself, when it differs only in numbers and white
space (_gist: a check run again with nothing changed in between).  A call that waits ("sleep" in its arguments) needs
the very same result.  A user message that only says to go on ("continue", "ok") or holds only <system-reminder>
blocks (Claude Code's text beside its tool results) does not start the count over; any other one does.  Calls made
in one turn count once (a batch that reads one file twice is no loop).  The run in a row stays as it was (repeated
and note): it fires whatever the results, and when both fire its note is the one written.  Still a note only:
nothing is blocked and the client's conversation is not changed.

    windowed(messages, times) -> (tool names, times) or None
    window_note(names, times)
    window_setting(value)    the run config's "loop_guard": {"window": false} turns the window off (-> 0, else TIMES)
"""
from __future__ import annotations

import json
import re

REPEATS = 4     # the same call this many times in a row: a loop (2 at the least; a single call is no repeat)
WINDOW = 12     # the last this many tool calls since the user's last message ...
TIMES = 3       # ... hold a call of the last turn (same arguments, same result) this many times: a loop


def args_key(value) -> str:
    """A call's arguments in one form: a client's JSON string and the parser's dict compare equal (Maya's args_key)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _calls(message: dict) -> tuple:
    """An assistant message's tool calls as ((name, args_key), ...), in their order."""
    out = []
    for c in message.get("tool_calls") or []:
        fn = c.get("function") if isinstance(c, dict) and isinstance(c.get("function"), dict) else c
        if not isinstance(fn, dict):
            continue
        out.append((str(fn.get("name")), args_key(fn.get("arguments"))))
    return tuple(out)


def repeated(messages: list[dict], repeats: int = REPEATS) -> tuple[str, int] | None:
    """The template messages of a request: does it answer tool results of a call (or a batch of calls) that the model
    made `repeats` times or more in a row?  -> (the calls' names, how many times) or None."""
    if not repeats or not messages:
        return None
    last = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "assistant"), None)
    if last is None or not messages[last].get("tool_calls"):
        return None
    if not any(m.get("role") == "tool" for m in messages[last + 1:]):
        return None                                   # the request does not answer that call's results
    key, times = _calls(messages[last]), 0
    if not key:
        return None
    for m in reversed(messages[:last + 1]):
        if m.get("role") != "assistant":
            continue                                  # tool results, and the client's own messages between
        if not m.get("tool_calls") or _calls(m) != key:
            break
        times += 1
    if times < repeats:
        return None
    names = ", ".join(dict.fromkeys(name for name, _ in key))
    return names, times


def note(names: str, times: int) -> str:
    return (f"[Note from the server, not the user: you have made the same tool call ({names}, with the same "
            f"arguments) {times} times in a row. Repeating it will not give a different result. Try something "
            "different, or stop and say what is blocking you.]")


def _text(content) -> str:
    """A message's content as text: a string as is, a list of content parts by its text parts (Maya 989ee69)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


GO_ON = re.compile(r"(?i)\s*(please\s+)?(continue|proceed|go on|go ahead|keep going|carry on|resume|next|ok(ay)?|"
                   r"yes|y|sure|do it)\s*[.!]*\s*")
REMINDER = re.compile(r"(?s)<system-reminder>.*?</system-reminder>")   # the rule; strip_reminders runs it in linear time
_OPEN, _CLOSE = "<system-reminder>", "</system-reminder>"


def strip_reminders(text: str) -> tuple[bool, str]:
    """(whether the text holds a whole <system-reminder> block, the text without those blocks) - what REMINDER.search /
    REMINDER.sub give, in one pass.  The regex retries its lazy scan at every opener, so a message of many openers and
    no closer took quadratic time on every chat request.  Here the leftmost opener's block ends at the first closer
    after it (the lazy match), and an opener with no closer after it means no later one has one either."""
    out, i, found = [], 0, False
    while True:
        a = text.find(_OPEN, i)
        if a < 0:
            break
        b = text.find(_CLOSE, a + len(_OPEN))
        if b < 0:
            break
        out.append(text[i:a])
        i = b + len(_CLOSE)
        found = True
    if not found:
        return False, text
    out.append(text[i:])
    return True, "".join(out)


def no_news(message: dict) -> bool:
    """A user message that says nothing new: only "continue" / "ok" and the like (Maya's go_on: a user's "continue"
    set GLM-5.3 off on the same loop again), or only <system-reminder> blocks (Claude Code sends them beside its tool
    results).  Such a message does not start the window's count over."""
    text = _text(message.get("content"))
    if GO_ON.fullmatch(text):
        return True
    found, rest = strip_reminders(text)
    return found and not rest.strip()


def _gist(result: str) -> str:
    """A result without its numbers, white space and control characters (after Maya's _gist, which also dropped all
    non-ASCII: two different results in CJK text then counted as the same)."""
    return re.sub(r"\d+|[\s\x00-\x1f\x7f-\x9f]+", "", result)


def _window_calls(messages: list[dict], window: int = WINDOW) -> list[tuple]:
    """(turn, name, args_key, result, gist) of the last `window` answered tool calls since the user's last message, in
    order.  Only those results are read (json.dumps, _gist): a long agent session holds megabytes of older ones."""
    calls, pending, turn = [], [], 0
    for m in messages:
        role = m.get("role")
        if role == "user" and not no_news(m):
            calls, pending = [], []
        elif role == "assistant":
            turn += 1
            pending = list(_calls(m))
        elif role == "tool" and pending:             # results answer the calls in their order
            name, args = pending.pop(0)
            calls.append((turn, name, args, m.get("content")))
    out = []
    for turn, name, args, content in calls[-window:] if window > 0 else []:
        result = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
        out.append((turn, name, args, result, _gist(result)))
    return out


def _times(window: list[tuple], at: int) -> int:
    """In how many turns of `window` the call window[at] was made with the same result - or, in the turns right
    before it, one after another with nothing else in between, with a result the same but for numbers (Maya's _times).
    A call that waits ("sleep") needs the very same result."""
    c = window[at]
    exact = len({x[0] for x in window if x[1:4] == c[1:4]})
    if "sleep" in c[2]:
        return exact
    run = set()
    for x in reversed(window[:at + 1]):
        if x[1:3] != c[1:3] or x[4] != c[4]:
            break
        run.add(x[0])
    return max(exact, len(run))


def windowed(messages: list[dict], times: int = TIMES, window: int = WINDOW) -> tuple[str, int] | None:
    """The template messages of a request that answers the results of the model's last turn: does the last `window`
    calls since the user's last message hold one of that turn's calls `times` times or more (_times)?
    -> (the names of the calls that do, the most times) or None."""
    if not times or not messages:
        return None
    last = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "assistant"), None)
    if last is None or not messages[last].get("tool_calls"):
        return None
    if not any(m.get("role") == "tool" for m in messages[last + 1:]):
        return None
    calls = _window_calls(messages, window)
    turn = sum(1 for m in messages[:last + 1] if m.get("role") == "assistant")
    hits = {}
    for k, c in enumerate(calls):
        if c[0] == turn:
            t = _times(calls, k)
            if t >= times:
                hits[c[1]] = max(t, hits.get(c[1], 0))
    if not hits:
        return None
    return ", ".join(hits), max(hits.values())


def window_note(names: str, times: int) -> str:
    return (f"[Note from the server, not the user: you have made the same tool call ({names}, with the same "
            f"arguments) {times} times in your last few tool calls, and it gave the same result each time. Repeating "
            "it will not give a different result. Try something different, or stop and say what is blocking you.]")


def window_setting(value) -> int:
    """The run config's "loop_guard" -> the times that make a loop over the window (0: off).  setting() checks it."""
    if value is False or (isinstance(value, dict) and value.get("window") is False):
        return 0
    return TIMES


def setting(value) -> int:
    """The run config's "loop_guard" -> the repeats that make a loop (0: off).  ValueError for a bad value."""
    if value is None or value is True:
        return REPEATS
    if value is False:
        return 0
    if isinstance(value, dict) and set(value) <= {"repeats", "window"} and isinstance(value.get("window", True), bool):
        n = value.get("repeats", REPEATS)
        if isinstance(n, int) and not isinstance(n, bool) and n >= 2:
            return n
    raise ValueError('"loop_guard" must be false, true or {"repeats": N, "window": false} with N 2 or more (both '
                     'keys optional), not ' + repr(value))
