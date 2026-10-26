"""Portable input paths shared by the HXQ3K-lite tools."""
import os
from pathlib import Path

Q3K_GGUF = os.environ.get("LITE_Q3K_GGUF", "heretic-Q3_K-exps.gguf")
# Optional: use extracted tables plus small shard copies. Without these settings,
# read the complete checkpoint shards through v41_calib as usual.
ENGRAM = {}
ENGRAM_SMALL = {}
small = os.environ.get("LITE_ENGRAM_SMALL_DIR")
for shard, layer in ((47, 1), (48, 14)):
    name = f"model-{shard:05d}-of-00048.safetensors"
    table = os.environ.get(f"LITE_ENGRAM_L{layer}")
    if table:
        ENGRAM[name] = table
    if small:
        if not table:
            raise ValueError(f"LITE_ENGRAM_SMALL_DIR requires LITE_ENGRAM_L{layer}")
        ENGRAM_SMALL[name] = str(Path(small) / ("engram-small-" + name))
