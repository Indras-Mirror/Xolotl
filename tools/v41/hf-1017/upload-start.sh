#!/usr/bin/env bash
# Start (or resume) the Hugging Face upload detached from this terminal.  Log: ~/AI/Strata-Backup/v41-dev/hf-1017/upload.log
# A supervisor loop reruns hf_upload.py until it reports ALL DONE (each run skips what is already on the Hub; xet
# dedup skips chunks already sent).  Pause with upload-stop.sh (stops the loop first).
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W=~/AI/Strata-Backup/v41-dev/hf-1017
mkdir -p $W
if [ -f $W/loop.pid ] && kill -0 "$(cat $W/loop.pid)" 2>/dev/null; then echo "already running (loop pid $(cat $W/loop.pid))"; exit 0; fi
# hf_xet's adaptive controller ramped to ~27 parallel 64 MB uploads on a home uplink: requests timed out and were
# retried (~1/3 of the bandwidth wasted, 2026-10-12) -> a fixed small concurrency.  One request then hung 40+ min with
# no timeout -> explicit read / connect timeouts (humantime strings).
export HF_XET_FIXED_UPLOAD_CONCURRENCY="${HF_XET_FIXED_UPLOAD_CONCURRENCY:-3}"
export HF_XET_CLIENT_READ_TIMEOUT="${HF_XET_CLIENT_READ_TIMEOUT:-300s}"
export HF_XET_CLIENT_CONNECT_TIMEOUT="${HF_XET_CLIENT_CONNECT_TIMEOUT:-30s}"
setsid nohup bash -c "
  echo \$\$ > $W/loop.pid
  until python3 -u '$here/hf_upload.py' $*; do
    echo \"\$(date +%T) hf_upload.py exited with an error - restarting in 60 s\"
    sleep 60
  done
  rm -f $W/loop.pid
" >> $W/upload.log 2>&1 < /dev/null &
sleep 1
echo "started (loop pid $(cat $W/loop.pid 2>/dev/null)) - tail -f $W/upload.log"
