"""DSpine/DFlash2 offline acceptance, step 1 (GPU): greedy answers from OUR 27B (as served, thinking off), then each
prompt+answer teacher-forced once with XOLOTL_TAP_DUMP on, so the tap rows of every position land in taps.bin."""
import json, os, sys, threading, time
sys.path.insert(0, "/home/mal/AI/Xolotl-dn")
W = "/home/mal/v41-work/dspine-20261109"
TF = os.environ.get("TAPF", "taps.bin")
os.environ["XOLOTL_TAP_DUMP"] = W + "/" + TF
os.environ["XOLOTL_TAP_LAYERS"] = os.environ.get("TAPL", "1,5,16,19,31,33,46,47,61")      # DSpine 1,16,31,46,61 + DFlash2 5,19,33,47,61
if os.path.exists(W + "/" + TF): os.remove(W + "/" + TF)
from serve.server import StrataEngine
sys.path.insert(0, "/home/mal/AI/Xolotl-dn/tools")
import strata_tokenizer as ST
from pathlib import Path
tp = Path("/home/mal/AI/Xolotl-dn/data/packs/qwen3.8-27b/tokenizer")
vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
toks = [None] * len(vocab)
for t, i in vocab.items(): toks[i] = t
tcfg = json.loads((tp / "tokenizer.json").read_text(encoding="utf-8"))
tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"), json.loads((tp / "token_type.json").read_text()),
                   tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)
P = json.load(open(W + "/prompts.json"))
MAXNEW = int(sys.argv[1]) if len(sys.argv) > 1 else 640
eng = StrataEngine("/media/Crucial1TB/xolotl-build/dn/xolotl-dense",
                   ["--native", "/media/Crucial1TB/Xolotl/qwen3.8-27b/Swift-1.5-27B-GSQ-RCO/d0xin-unc-27b-IQ3_S-mtp-q6embd.gguf",
                    "--context", "8192", "--kv", "k8v4"], cwd="/home/mal/AI/Xolotl-dn", log=W + "/engine.log")
def ids_of(p):
    return tok.encode("<|im_start|>user\n" + p["text"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      parse_special=True)
out = json.load(open(W + "/seqs.json")) if os.path.exists(W + "/seqs.json") else []
for p in ([] if out else P):
    ids = ids_of(p)
    t = time.time()
    gen = [x for x in eng.generate(ids, MAXNEW, {"temperature": 0.0}, threading.Event()) if x is not None]
    out.append({"name": p["name"], "prompt": ids, "out": gen})
    print("%s: prompt %d, generated %d in %.1f s" % (p["name"], len(ids), len(gen), time.time() - t), flush=True)
json.dump(out, open(W + "/seqs.json", "w"))
if os.path.exists(W + "/" + TF): os.remove(W + "/" + TF)   # the generation requests' rows are not needed
for o in out:                                    # teacher-forced: prompt + answer read as one prompt
    seq = o["prompt"] + o["out"]
    for _ in eng.generate(seq, 1, {"temperature": 0.0}, threading.Event()): pass
    print("%s: tapped %d rows" % (o["name"], len(seq)), flush=True)
json.dump(out, open(W + "/seqs.json", "w"))
eng.close() if hasattr(eng, "close") else None
print("done", flush=True)
