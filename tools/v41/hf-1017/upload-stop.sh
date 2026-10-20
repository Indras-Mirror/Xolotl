#!/usr/bin/env bash
# Pause the upload: stop the supervisor loop, then the uploader (by PID only).  upload-start.sh resumes.
W=~/AI/Strata-Backup/v41-dev/hf-1017
l=$(cat $W/loop.pid 2>/dev/null)
if [ -n "$l" ] && kill -0 "$l" 2>/dev/null; then kill "$l" && echo "stopped loop (pid $l)"; fi
rm -f $W/loop.pid
pid=$(cat $W/upload.lock 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && grep -q hf_upload.py "/proc/$pid/cmdline" 2>/dev/null; then
    kill "$pid" && echo "stopped upload (pid $pid)"
else
    echo "no uploader running"
fi
