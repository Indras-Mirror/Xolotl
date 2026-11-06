#!/usr/bin/env bash
# load.sh - checks the machine, then loads the two Xolotl Direct modules (run with sudo; nothing loads them at boot).
#   sudo kernel/xolotl-direct/load.sh          load
#   sudo kernel/xolotl-direct/load.sh unload   unload (stop Xolotl first)
set -eu
cd "$(dirname "$0")"
if [ "${1:-}" = unload ]; then rmmod xolp2p 2>/dev/null || true; rmmod p2pgate 2>/dev/null || true; echo "unloaded"; exit 0; fi
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
grep -q '^CONFIG_PCI_P2PDMA=y' "/boot/config-$(uname -r)" 2>/dev/null || echo "warning: CONFIG_PCI_P2PDMA=y not found in /boot/config-$(uname -r)"
bar=$(nvidia-smi -q 2>/dev/null | awk '/BAR1 Memory Usage/{f=1} f&&/Total/{print $3; exit}')
[ -n "$bar" ] && [ "$bar" -ge 4096 ] || { echo "BAR1 is ${bar:-?} MiB: turn on Above 4G Decoding + Re-Size BAR in the BIOS"; exit 1; }
[ -f p2pgate/p2pgate.ko ] && [ -f xolp2p/xolp2p.ko ] || { echo "build first: make -C $(pwd)"; exit 1; }
lsmod | grep -q '^p2pgate ' || insmod p2pgate/p2pgate.ko
lsmod | grep -q '^xolp2p ' || insmod xolp2p/xolp2p.ko
ls -l /dev/p2pgate /dev/xolp2p
echo "loaded (BAR1 ${bar} MiB). Your user needs group 'disk' and the udev rule in docs/XOLOTL_DIRECT.md."
