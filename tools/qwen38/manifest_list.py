#!/usr/bin/env python3
"""Print "name<TAB>size<TAB>algo<TAB>hash" for every file in the fetch manifest.

Largest first, so the long files start while the connection is fresh and the small metadata
files finish at the end.

Two hash algorithms appear: the weight shards are LFS objects with a 64-hex sha256, while the
small text/metadata files carry HF's `oid`, which is a GIT BLOB hash: sha1("blob <len>\0"+content),
40 hex.  That is NOT sha1sum of the bytes - use `git hash-object <file>` to check it.  Both verify
the bytes on disk; they just need different tools, so the algorithm travels with the hash.
"""
import json
import sys

m = json.load(open(sys.argv[1]))
for name, v in sorted(m.items(), key=lambda kv: -kv[1]["size"]):
    print(f"{name}\t{v['size']}\t{v['algo']}\t{v['hash']}")
