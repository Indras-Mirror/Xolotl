# Xolotl Direct: NVMe straight into VRAM on a GeForce card

> **DRAFT (2026-11-07), not shipped yet.** Everything below is measured on one box (RTX 4090, Linux 7.0, driver
> 595.91, Samsung PM9A1 NVMe). The "Before it ships" list at the end is what still has to happen.

When a model is bigger than your RAM, Xolotl reads the experts it is missing from the NVMe drive while it decodes.
Normally those bytes take the long way: NVMe -> a bounce buffer in RAM -> pinned staging -> PCIe -> VRAM. That is four
trips over the RAM bus, and on a DDR4 box the RAM bus is the wall.

**Xolotl Direct** lets the NVMe drive write the expert straight into the graphics card's memory, over PCIe, without
touching system RAM. It runs on the normal NVIDIA driver on a GeForce card: no driver patch, no datacenter card, no
GPUDirect Storage licence. It is opt-in, and if anything is missing Xolotl says so once and uses the normal path.

## What it does today

| | normal path | Xolotl Direct |
|---|---|---|
| One 18 MiB expert, NVMe -> VRAM | ~11 ms | **3.1 ms** |
| 64 MiB read, best of 3 | 4.61 GB/s | **6.31 GB/s** (the drive's own limit) |
| System RAM traffic per read | 4 passes over the bus | **none** |
| Bytes | - | every blob checked against the normal read: **0 differences** in thousands |
| Output | - | **same tokens** as the normal path |

In the DeepSeek-V4.1 engine (V4, 4090, 8K prompt, 512 tokens, two rounds with identical VRAM):

| | ms per decode pass | tok/s |
|---|---|---|
| normal reads | 116.5 | 25.36 |
| Xolotl Direct | 116.0 | 25.48 |
| Xolotl Direct + file lookahead | **113.0** | **26.14 (+3.1 %)** |

Honest note: on its own, the direct path saves ~1.7 ms of file waiting per pass but barely moves the pass, because
those reads are already overlapped with other work. Its gain comes from what it enables: a VRAM stash that the
lookahead fills a layer early with no RAM-bus cost.

## What you need

- **Linux** with `CONFIG_PCI_P2PDMA=y` (most distro kernels; check `grep P2PDMA /boot/config-$(uname -r)`).
  Tested on 7.0. Windows: no.
- **An NVIDIA RTX card** (tested: RTX 4090, Ada) on the normal driver (tested: 595.91, the proprietary module).
- **Resizable BAR on.** In the BIOS: Above 4G Decoding ON, Re-Size BAR ON, CSM OFF. Check after booting:
  `nvidia-smi -q | grep -A3 BAR1` should show the card's whole memory (24576 MiB or more), not 256 MiB.
- **The model on an NVMe drive** served by the normal `nvme` driver (nothing is unbound), on a filesystem that
  supports `FIEMAP` and `O_DIRECT` (tested: ntfs3, ext4).
- **Kernel headers** for your running kernel (`linux-headers-$(uname -r)`) to build two small modules.
- **Secure Boot off**, or sign the two modules with your own MOK key (see below).

## Set it up

1. **Build the two kernel modules** (both GPL, a few hundred lines each):
   - `p2pgate`: asks the NVIDIA driver to pin a window of VRAM and keep it pinned (`nvidia_p2p_get_pages_persistent`).
   - `xolp2p`: registers that window with the kernel's PCI peer-to-peer support, so the NVMe drive may write into it.

   ```bash
   cd kernel/xolotl-direct        # (path when shipped)
   make                           # builds p2pgate.ko and xolp2p.ko against your running kernel
   ```

2. **Secure Boot only:** sign both modules
   (`/usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 MOK.priv MOK.der p2pgate.ko`, the same for xolp2p.ko)
   after enrolling your key with `mokutil --import MOK.der`.

3. **Load them**, every boot (nothing loads them automatically, on purpose):

   ```bash
   sudo insmod p2pgate.ko && sudo insmod xolp2p.ko
   ```

4. **Let your user open them.** The modules create `/dev/p2pgate` and `/dev/xolp2p`, root-only by default. This udev
   rule gives them to group `disk`, which can already read raw disks, so it grants nothing new:

   ```
   # /etc/udev/rules.d/99-xolotl-direct.rules
   KERNEL=="p2pgate", SUBSYSTEM=="misc", GROUP="disk", MODE="0660"
   KERNEL=="xolp2p",  SUBSYSTEM=="misc", GROUP="disk", MODE="0660"
   ```

   Xolotl Direct reads the model's own partition raw at the file's extents, so the user that runs Xolotl needs to be
   in group `disk` (`sudo usermod -aG disk $USER`, then log in again). Group `disk` can read every disk on the
   machine; if that is too much for your setup, run the server as a dedicated user.

5. **Self-test** (reads 64 MiB of your model both ways and compares every byte):

   ```bash
   xolotl-direct-selftest /path/to/model.gguf
   ```

## Use it

```bash
XOLOTL_P2P=1 XOLOTL_LOOKAHEAD=2 XOLOTL_LA_SPEC=2  <your usual Xolotl serve command>
```

| switch | default | what |
|---|---|---|
| `XOLOTL_P2P=1` | off | file-tier experts NVMe -> VRAM |
| `XOLOTL_LOOKAHEAD=2` + `XOLOTL_LA_SPEC=2` | off | guess the next layer's file reads and fetch them a layer early into the VRAM stash (the +3 %) |
| `XOLOTL_P2P_SLOTS` / `XOLOTL_P2P_SPEC_SLOTS` | 8 / 8 | VRAM window size, in experts (~18 MiB each for V4) |
| `XOLOTL_P2P_VERIFY=1` | off | check every direct read against a normal read (slow; for testing) |

At startup the log says `STRATA_P2P on - NVMe -> VRAM ring, ...` (the engine's internal names still say STRATA; every
switch takes the XOLOTL_ name). If it says `STRATA_P2P unavailable (...)`, the reason follows and Xolotl carries on
with the normal path.

**Turn it off:** unset `XOLOTL_P2P`. To remove the modules: stop Xolotl, then `sudo rmmod xolp2p p2pgate`.

## Safety

- The modules do nothing until Xolotl opens them; they never touch the NVMe driver or the filesystem.
- Both check a magic number before freeing anything (they refuse and warn instead of freeing a foreign pointer), and
  both must unload cleanly; the staged test script refuses to continue if they do not.
- One kernel-side registration (the VRAM window's page structs) lasts until the next reboot. It is harmless and is
  reused on the next load.
- During development an early version of `xolp2p` crashed the kernel (a NULL pointer on release). That bug is fixed
  and the magic checks were added because of it. Load the modules on a machine where a reboot is acceptable the
  first time.

## Before it ships (the to-do list)

- [ ] A clean decode win in the default serve config, confirmed over more rounds and prompts.
- [ ] The file lookahead also on the normal path, so users without the modules get its gain too.
- [ ] Module sources and the user-space library moved into the public tree (`kernel/xolotl-direct/`), a CMake option
      (`-DXOLOTL_DIRECT=ON`) instead of the test build script, and the self-test as a shipped binary.
- [ ] A one-command check script (BAR1 size, P2PDMA, modules, device permissions, filesystem, self-test).
- [ ] Tested on a second card (RTX 30) and a second drive.
- [ ] Prior-art check before any "first" claim.
- [ ] Faster model loading through the same path.

## README section (to paste when it ships)

> - **Xolotl Direct (Linux, opt-in): NVMe straight into VRAM on a GeForce card.** File-tier experts skip system RAM:
>   one expert 11 -> 3.1 ms, no RAM-bus traffic, the same output. With the layer-ahead stash: +3 % decode on
>   DeepSeek-V4.1. Needs Resizable BAR and two small kernel modules. [Setup](docs/XOLOTL_DIRECT.md)
