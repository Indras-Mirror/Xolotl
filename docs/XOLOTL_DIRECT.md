# Xolotl Direct: NVMe straight into VRAM on a GeForce card

> **Experimental, opt-in (7 Nov 2026).** Measured on one box so far (RTX 4090, Linux 7.0, driver 595.91, Samsung
> PM9A1 NVMe). Off by default and off in every build that does not ask for it. Reports from other cards and drives
> are very welcome (open an issue with the self-test output).

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

1. **Build the two kernel modules** (both GPL, a few hundred lines each, in `kernel/xolotl-direct/`):
   - `p2pgate`: asks the NVIDIA driver to pin a window of VRAM and keep it pinned (`nvidia_p2p_get_pages_persistent`).
   - `xolp2p`: registers that window with the kernel's PCI peer-to-peer support, so the NVMe drive may write into it.

   ```bash
   make -C kernel/xolotl-direct        # needs linux-headers-$(uname -r); finds the NVIDIA Module.symvers itself
                                       # (DKMS installs) - else: make ... NV_SYMVERS=/path/to/Module.symvers
   ```

2. **Secure Boot only:** sign both `.ko` files with your own MOK key
   (`/usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 MOK.priv MOK.der <module>.ko`) after enrolling the
   key with `mokutil --import MOK.der`.

3. **Build Xolotl with it.** The user-space side talks to the NVIDIA driver through its SDK headers, so it needs
   NVIDIA's open-gpu-kernel-modules source **of the driver version you have installed**
   (`cat /proc/driver/nvidia/version`):

   ```bash
   git clone --depth 1 --branch 595.91.07 https://github.com/NVIDIA/open-gpu-kernel-modules ~/src/nvidia-ogkm
   cmake -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_DSV41=ON -DXOLOTL_DIRECT=ON \
         -DXOLOTL_DIRECT_NVSRC=$HOME/src/nvidia-ogkm      # plus your usual options
   cmake --build build --target ds41_generate xolotl-direct-selftest
   ```

   After a driver update, check out the matching tag and rebuild (the library refuses a driver it was not built for).

4. **Let your user open the devices** (once). `/dev/p2pgate` and `/dev/xolp2p` are root-only by default; the rule in
   `kernel/xolotl-direct/99-xolotl-direct.rules` gives them to group `disk`, which can already read raw disks, so it
   grants nothing new:

   ```bash
   sudo cp kernel/xolotl-direct/99-xolotl-direct.rules /etc/udev/rules.d/ && sudo udevadm control --reload
   sudo usermod -aG disk $USER        # then log in again
   ```

   Xolotl Direct reads the model's own partition raw, at the file's extents, so the user that runs Xolotl must be in
   group `disk`. Group `disk` can read every disk on the machine; if that is too much for your setup, run the server
   as a dedicated user.

5. **Load the modules**, every boot (nothing loads them automatically, on purpose). The script checks Resizable BAR
   and P2PDMA first:

   ```bash
   sudo kernel/xolotl-direct/load.sh            # sudo kernel/xolotl-direct/load.sh unload  to remove them
   ```

6. **Self-test** (reads 64 MiB of your model both ways and compares every byte):

   ```bash
   build/xolotl-direct-selftest /path/to/model.gguf
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

**Turn it off:** unset `XOLOTL_P2P`. To remove the modules: stop Xolotl, then `sudo kernel/xolotl-direct/load.sh unload`.

## Safety

- The modules do nothing until Xolotl opens them; they never touch the NVMe driver or the filesystem.
- Both check a magic number before freeing anything (they refuse and warn instead of freeing a foreign pointer), and
  both must unload cleanly; the staged test script refuses to continue if they do not.
- One kernel-side registration (the VRAM window's page structs) lasts until the next reboot. It is harmless and is
  reused on the next load.
- During development an early version of `xolp2p` crashed the kernel (a NULL pointer on release). That bug is fixed
  and the magic checks were added because of it. Load the modules on a machine where a reboot is acceptable the
  first time.

## Still to do

- [ ] Confirm the decode gain over more rounds and prompts, and on a second card (RTX 30) and drive.
- [ ] The file lookahead on the normal path too, so users without the modules get its gain.
- [ ] Faster model loading through the same path.
- [ ] A prior-art check before any "first" claim.

## README line

> - **Xolotl Direct (Linux, opt-in): NVMe straight into VRAM on a GeForce card.** File-tier experts skip system RAM:
>   one expert 11 -> 3.1 ms, no RAM-bus traffic, the same output. With the layer-ahead stash: +3 % decode on
>   DeepSeek-V4.1. Needs Resizable BAR and two small kernel modules. [Setup](docs/XOLOTL_DIRECT.md)
