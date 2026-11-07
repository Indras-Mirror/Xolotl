#!/bin/bash
# llama-swap cmd for Xolotl DeepSeek-V4.1: CONFIG PORT.  As /root/strata/start.sh: wait until the previous engine
# (Strata or ds41) has exited and released its pinned RAM / VRAM, then start the server.
C=$1; P=$2
for i in $(seq 1 240); do pgrep -x strata >/dev/null || pgrep -x ds41_generate >/dev/null || break; sleep 0.5; done
cd /root/xolotl
exec /root/xolotl/.venv/bin/python /root/xolotl/tools/v41/serve/serve_ds41.py --engine strata --config "$C" --host 127.0.0.1 --port "$P"
