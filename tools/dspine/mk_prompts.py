import json
code = '''def load_rows(path):
    rows = []
    f = open(path)
    for line in f.readlines():
        if line.strip() == "":
            continue
        parts = line.split(",")
        rows.append({"id": int(parts[0]), "name": parts[1], "score": float(parts[2])})
    f.close()
    best = None
    for r in rows:
        if best == None or r["score"] > best["score"]:
            best = r
    return rows, best
'''
P = [
 {"name": "code_py", "text": "Write a Python class LRUCache with get(key) and put(key, value) in O(1), using a dict and a doubly linked list (no OrderedDict). Include type hints, docstrings and a short pytest test file."},
 {"name": "code_edit", "text": "Refactor this function: use a context manager and the csv module, skip malformed rows with a warning via logging, return a dataclass instead of dicts, and keep the same behaviour otherwise. Show the full new code, then list the changes.\n\n```python\n" + code + "```"},
 {"name": "agent", "text": "You are a coding agent in a Linux repo. The test `tests/test_api.py::test_pagination` fails with `AssertionError: assert 21 == 20`. Explain step by step which files you would inspect and which shell commands you would run (grep, pytest -x, git log -p), then write the most likely fix for an off-by-one in a function `paginate(items, page, per_page)` and a regression test."},
 {"name": "explain", "text": "Explain how speculative decoding works in LLM inference: the draft model, verification in one forward pass, acceptance, why the output distribution is unchanged, and what limits the speed-up on a memory-bandwidth-bound GPU. Use headings and a small worked example."},
 {"name": "math", "text": "A tank holds 1,200 litres. Pipe A fills it in 6 hours, pipe B in 4 hours, and a drain empties it in 8 hours. Starting empty with all three open, after 2 hours pipe B is closed. How long in total until the tank is full? Show every step, then verify the answer."},
 {"name": "chat", "text": "I'm moving from Melbourne to Brisbane for work next month with a dog and a lot of computer gear. Give me a practical checklist: timeline, packing the PC and GPUs safely, the dog, utilities and internet, and things people usually forget."},
]
json.dump(P, open("/home/mal/v41-work/dspine-20261109/prompts.json", "w"), indent=1)
