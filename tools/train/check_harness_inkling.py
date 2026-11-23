#!/usr/bin/env python3
"""Gate for the Inkling-2-Mini harness set (harness-inkling/).

Checks, on every window:
  * ids are the conversation AFTER the prefix: the window starts at a `<|message_user|>` header;
  * decode round-trips: decode(ids) re-encodes to exactly ids (with the window's `plain` spans, when it has any);
  * the loss mask covers ONLY assistant turns: each masked run starts at `<|message_model|>` and ends after its
    `<|content_model_end_sampling|>`, contains exactly one end-of-sampling, and the token before it (when there is one)
    closes the previous prompt message.  A mask injected on a prompt token is built and MUST be rejected;
  * no window is longer than 3200 tokens;
  * stats.json agrees with the data, and heldout.txt is disjoint from the windows and counted.

usage: python3 check_harness_inkling.py [--dir DIR] [--max-tokens 3200] [--limit N] [--samples 2]
Exit 0 = pass.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness_build_inkling import HEADER, EOS, load_tokenizer, TOKDIR  # noqa: E402

USER = "<|message_user|>"
END = "<|end_message|>"
THINK, TEXT, TOOL_JSON = "<|content_thinking|>", "<|content_text|>", "<|content_invoke_tool_json|>"
MAX_REPORT = 12


class Bad(Exception):
    pass


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def check_mask(ids, mask, tok, wid):
    """Every masked run is exactly one assistant turn; every assistant turn that is masked is a whole one."""
    hdr, eos, end = tok.ids[HEADER], tok.ids[EOS], tok.ids[END]
    user = tok.ids[USER]
    rs = runs(mask)
    if not rs:
        raise Bad(f"{wid}: mask is empty")
    for i, j in rs:
        if ids[i] != hdr:
            raise Bad(f"{wid}: masked run at token {i} starts with {tok.decode(ids[i:i + 1])!r}, not "
                      f"<|message_model|> (a prompt token is masked)")
        if ids[j - 1] != eos:
            raise Bad(f"{wid}: masked run at {i}..{j} ends with {tok.decode(ids[j - 1:j])!r}, not "
                      f"<|content_model_end_sampling|>")
        body = ids[i:j]
        if body.count(eos) != 1:
            raise Bad(f"{wid}: masked run at {i}..{j} holds {body.count(eos)} end-of-sampling tokens (spans merged?)")
        # the block that follows the header must open with a content marker (a bare name before it is a tool's name)
        k = 1
        while k < len(body) and body[k] not in (tok.ids[THINK], tok.ids[TEXT], tok.ids[TOOL_JSON]):
            k += 1
        if k == len(body):
            raise Bad(f"{wid}: masked run at {i}..{j} has no thinking/text/tool block after its header")
        if i > 0 and ids[i - 1] not in (end, eos):
            raise Bad(f"{wid}: the token before masked run {i} is {tok.decode(ids[i - 1:i])!r}, not </end_message> "
                      f"(mask starts inside a prompt turn)")
    # a masked run must not swallow the window's prompt opener
    if mask[0] and ids[0] == user:
        raise Bad(f"{wid}: the prompt's <|message_user|> token is masked")
    return len(rs)


def mark_text(ids, mask, tok):
    """The decoded window with the mask boundaries marked."""
    parts, i = [], 0
    while i < len(ids):
        if mask[i]:
            j = i
            while j < len(ids) and mask[j]:
                j += 1
            parts.append("\n<<<LOSS %d tokens>>>\n" % (j - i) + tok.decode(ids[i:j]) + "\n<<<END LOSS>>>\n")
            i = j
        else:
            j = i
            while j < len(ids) and not mask[j]:
                j += 1
            parts.append(tok.decode(ids[i:j]))
            i = j
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=3200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    d = Path(a.dir)
    fails, checked, tokens, loss, runs_total = [], 0, 0, 0, 0

    def fail(msg):
        fails.append(msg)
        if len(fails) <= MAX_REPORT:
            print("FAIL " + msg)

    tok = load_tokenizer(TOKDIR)
    pref = json.loads((d / "prefix.json").read_text())
    stats = json.loads((d / "stats.json").read_text())
    heldout = set((d / "heldout.txt").read_text().split()) if (d / "heldout.txt").exists() else set()

    if pref["n"] != len(pref["tokens"]):
        fail(f"prefix.json: n {pref['n']} != {len(pref['tokens'])} tokens")
    if pref["tokens"][0] != tok.ids["<|message_system|>"]:
        fail("prefix.json: does not start with the tool-declare system block")
    if len(heldout) != stats.get("convs_heldout"):
        fail(f"heldout.txt has {len(heldout)} ids, stats.convs_heldout is {stats.get('convs_heldout')}")

    lines = (d / "windows.jsonl").read_text().splitlines()
    if stats.get("windows") != len(lines):
        fail(f"stats.windows {stats.get('windows')} != {len(lines)} lines")
    by_prod, convs, samples = {}, set(), []
    for line in lines:
        w = json.loads(line)
        wid = w.get("id")
        ids, mask = w.get("ids") or [], w.get("mask") or []
        if len(ids) != w.get("n") or len(mask) != w.get("n"):
            fail(f"{wid}: n {w.get('n')} != len(ids) {len(ids)} / len(mask) {len(mask)}")
            continue
        if any(m not in (0, 1) for m in mask):
            fail(f"{wid}: mask has a value other than 0/1")
            continue
        if sum(mask) != w.get("n_loss"):
            fail(f"{wid}: n_loss {w.get('n_loss')} != sum(mask) {sum(mask)}")
        if not ids:
            fail(f"{wid}: empty window")
            continue
        if ids[0] != tok.ids[USER]:
            fail(f"{wid}: window does not start at <|message_user|> (prefix not removed?)")
        if len(ids) > a.max_tokens:
            fail(f"{wid}: {len(ids)} tokens > {a.max_tokens}")
        text = tok.decode(ids)
        back = tok.encode(text, parse_special=True, plain=[tuple(x) for x in w.get("plain") or []])
        if back != ids:
            fail(f"{wid}: decode does not round-trip ({len(ids)} ids -> {len(back)})")
        try:
            nr = check_mask(ids, mask, tok, wid)
            runs_total += nr
        except Bad as exc:
            fail(str(exc))
        if stats.get("heldout_only"):   # an eval set (builder --heldout-only): every window must be held out
            if w.get("conv") not in heldout:
                fail(f"{wid}: eval set (--heldout-only) holds a window of a training conversation {w.get('conv')}")
        elif w.get("conv") in heldout:
            fail(f"{wid}: conversation {w.get('conv')} is held out but has a training window")
        convs.add(w.get("conv"))
        by_prod[w.get("producer")] = by_prod.get(w.get("producer"), 0) + 1
        checked += 1
        tokens += len(ids)
        loss += sum(mask)
        if len(samples) < a.samples and sum(mask) >= 256:
            samples.append(w)
        if a.limit and checked >= a.limit:
            break

    # the mask check must reject a prompt token: inject one on the first sample (or the first window)
    victim = (samples[0] if samples else json.loads(lines[0])) if lines else None
    if victim is not None:
        bad = list(victim["mask"])
        bad[0] = 1
        try:
            check_mask(victim["ids"], bad, tok, victim["id"])
            fail("mask check did NOT reject a mask injected on the prompt's first token")
        except Bad:
            pass

    # stats consistency (a --limit run checks only the windows it read)
    if not a.limit:
        if stats.get("tokens") != tokens:
            fail(f"stats.tokens {stats.get('tokens')} != {tokens}")
        if stats.get("loss_tokens") != loss:
            fail(f"stats.loss_tokens {stats.get('loss_tokens')} != {loss}")
        if stats.get("turns_trained") != runs_total:
            fail(f"stats.turns_trained {stats.get('turns_trained')} != {runs_total} masked runs")
        src = stats.get("by_producer") or {}
        if sum(src.values()) != runs_total:
            fail(f"stats.by_producer sums to {sum(src.values())}, mask has {runs_total} runs")
        for p, c in by_prod.items():
            for name in (p or "").split(","):
                if name and name not in src:
                    fail(f"window producer {name!r} is not in stats.by_producer")

    if samples:
        with open(d / "samples.txt", "w") as f:
            for w in samples[:2]:
                f.write(f"===== window {w['id']}  conv {w.get('conv')}  producer {w.get('producer')}\n")
                f.write(f"n {w['n']}  n_loss {w['n_loss']}  ({100.0 * w['n_loss'] / w['n']:.1f} % loss)\n")
                f.write(mark_text(w["ids"], w["mask"], tok))
                f.write("\n")
    print(f"CHECKED {checked} windows, {tokens} tokens, {loss} loss tokens, {runs_total} assistant runs, "
          f"{len(convs)} conversations, prefix {pref['n']} tokens")
    if fails:
        print(f"FAILED: {len(fails)} problems" + (f" (first {MAX_REPORT} shown)" if len(fails) > MAX_REPORT else ""))
        return 1
    print(f"PASS: every window prefix-free, decode round-trips, mask = assistant turns only, all <= {a.max_tokens} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
