// xolp2p.c - Xolotl NVMe -> VRAM window (2026-11-07, Step C).  GPL, NO nvidia symbols (the licence check refuses a
// module that mixes GPL-only P2PDMA symbols with the proprietary driver's nvidia_p2p_* exports).
// The BAR1 bus range of a VRAM window that p2pgate HOLDs (pinned for third-party access) is handed to the kernel's
// PCI P2PDMA layer (ZONE_DEVICE pages) and mmap()ed in window order.  An O_DIRECT pread from the model's raw partition
// into that mapping is then DMA'd by the NVMe controller straight into VRAM, through the normal nvme driver.
// The whole of BAR1 is registered once per boot (device-managed on the GPU; stays until the nvidia driver unbinds),
// then each MAP claims exactly the pages of its window from the pool and returns the rest.
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/pci.h>
#include <linux/pci-p2pdma.h>
#include <linux/mutex.h>

struct xolp2p_map { __u64 bus, len; __s32 status; __u32 pages; __u64 bar_start, bar_len; };
#define XOLP2P_MAP _IOWR('x', 2, struct xolp2p_map)

#define XP_MAGIC 0x584f4c57u   /* "XOLW": release/mmap only ever touch a window this driver created */
struct xp_win { u32 magic; u64 bus, len; u32 npages; void **kva; bool mapped; };

// BAR1 is handed to P2PDMA in fixed 512 MiB blocks on a grid from the BAR start, only the blocks a window touches,
// each at most once per boot (registrations are device-managed on the GPU and outlive this module).  A fixed grid means
// two registrations can never partly overlap (memremap_pages WARNs "Conflicting mapping in same section" and fails),
// and a block costs ~8 MB of struct pages instead of ~512 MB for the whole 32 GB BAR (Resizable BAR on).  nvidia picks
// the window's place in BAR1 (it moves between runs), so the grid follows the window.
#define XP_BLOCK (512ull << 20)
#define XP_SPARE_MAX (2u << 20)   // pages parked while looking for the window's (8 GB of pool)

static DEFINE_MUTEX(xp_lock);
static struct pci_dev *gpu;

static struct pci_dev *find_gpu(void)
{
    struct pci_dev *d = NULL;
    while ((d = pci_get_device(PCI_VENDOR_ID_NVIDIA, PCI_ANY_ID, d)))
        if ((d->class >> 16) == PCI_BASE_CLASS_DISPLAY) return d;   // reference held
    return NULL;
}

static void win_free(struct xp_win *w)
{
    u32 i;
    if (!w) return;
    if (w->magic != XP_MAGIC) { WARN_ONCE(1, "xolp2p: refusing to free a non-window pointer\n"); return; }
    w->magic = 0;
    if (w->kva) {
        if (!w->mapped)   // a mapped window's pages went back to the pool at munmap (their last put)
            for (i = 0; gpu && i < w->npages; ++i) if (w->kva[i]) pci_free_p2pmem(gpu, w->kva[i], PAGE_SIZE);
        kvfree(w->kva);
    }
    kfree(w);
}

static long xp_map_ioctl(struct file *f, unsigned long arg)
{
    struct xolp2p_map r;
    struct xp_win *w;
    void **spare = NULL;
    u32 nspare = 0, cap, i, got = 0;
    int rc = 0;
    if (f->private_data) return -EBUSY;
    if (copy_from_user(&r, (void __user *) arg, sizeof(r))) return -EFAULT;
    if (!r.len || (r.bus | r.len) & (PAGE_SIZE - 1)) return -EINVAL;
    mutex_lock(&xp_lock);
    if (!gpu) gpu = find_gpu();
    if (!gpu) { rc = -ENODEV; goto out; }
    r.bar_start = pci_resource_start(gpu, 1); r.bar_len = pci_resource_len(gpu, 1);
    if (r.bus < r.bar_start || r.bus + r.len > r.bar_start + r.bar_len) { rc = -ERANGE; goto out; }
    {   // register every grid block the window touches that is not registered yet
        u64 b0 = round_down(r.bus - r.bar_start, XP_BLOCK), b1 = round_up(r.bus + r.len - r.bar_start, XP_BLOCK), b;
        for (b = b0; b < b1; b += XP_BLOCK) {
            const u64 sz = min_t(u64, XP_BLOCK, r.bar_len - b);
            struct dev_pagemap *pm = get_dev_pagemap(PHYS_PFN(r.bar_start + b));
            int ar;
            if (pm) { put_dev_pagemap(pm); continue; }   // already registered this boot
            pm = get_dev_pagemap(PHYS_PFN(r.bar_start + b + sz - 1));
            if (pm) {   // cannot happen on the grid; refuse rather than let memremap_pages WARN
                put_dev_pagemap(pm);
                pr_err("xolp2p: block 0x%llx is partly registered - refusing\n", r.bar_start + b);
                rc = -EEXIST; goto out;
            }
            ar = pci_p2pdma_add_resource(gpu, 1, (size_t) sz, b);
            pr_info("xolp2p: pci_p2pdma_add_resource(%s BAR1 block 0x%llx +0x%llx) = %d\n", pci_name(gpu), r.bar_start + b, sz, ar);
            if (ar) { rc = ar; goto out; }
        }
    }
    w = kzalloc(sizeof(*w), GFP_KERNEL);
    if (!w) { rc = -ENOMEM; goto out; }
    w->magic = XP_MAGIC; w->bus = r.bus; w->len = r.len; w->npages = (u32) (r.len >> PAGE_SHIFT);
    w->kva = kvcalloc(w->npages, sizeof(void *), GFP_KERNEL);
    cap = XP_SPARE_MAX;
    spare = kvcalloc(cap, sizeof(void *), GFP_KERNEL);
    if (!w->kva || !spare) { rc = -ENOMEM; win_free(w); goto out; }
    // claim pool pages one at a time; keep the window's, park the rest, give the rest back after
    for (;;) {
        void *k = pci_alloc_p2pmem(gpu, PAGE_SIZE);
        u64 b;
        if (!k) break;
        b = pci_p2pmem_virt_to_bus(gpu, k);
        if (b >= r.bus && b < r.bus + r.len && !w->kva[(b - r.bus) >> PAGE_SHIFT]) { w->kva[(b - r.bus) >> PAGE_SHIFT] = k; ++got; }
        else if (nspare < cap) spare[nspare++] = k;
        else { pci_free_p2pmem(gpu, k, PAGE_SIZE); break; }
        if (got == w->npages) break;
    }
    for (i = 0; i < nspare; ++i) pci_free_p2pmem(gpu, spare[i], PAGE_SIZE);
    kvfree(spare);
    r.pages = got;
    if (got != w->npages) {
        pr_err("xolp2p: window 0x%llx +0x%llx: only %u of %u pages in the pool\n", r.bus, r.len, got, w->npages);
        win_free(w); rc = -ENOSPC; goto out;
    }
    f->private_data = w;
    pr_info("xolp2p: window bus 0x%llx +0x%llx ready (%u pages)\n", r.bus, r.len, got);
out:
    mutex_unlock(&xp_lock);
    r.status = rc;
    if (copy_to_user((void __user *) arg, &r, sizeof(r))) return -EFAULT;
    return rc;
}

static int xp_mmap(struct file *f, struct vm_area_struct *vma)
{
    struct xp_win *w = f->private_data;
    unsigned long len = vma->vm_end - vma->vm_start, off;
    int rc = 0;
    if (!w || w->magic != XP_MAGIC || w->mapped) return -ENXIO;
    if ((vma->vm_flags & VM_MAYSHARE) != VM_MAYSHARE) return -EINVAL;   // as p2pmem_alloc_mmap: no private mappings
    if (vma->vm_pgoff || len != w->len) return -EINVAL;   // one whole-window mapping per MAP
    // Mirrors drivers/pci/p2pdma.c p2pmem_alloc_mmap (v7.0): count 1, insert, put -> the mapping owns the page and its
    // last put (munmap) runs p2pdma_folio_free = gen_pool_free + percpu_ref_put, which balances the ref
    // pci_alloc_p2pmem took for that page.
    for (off = 0; off < len; off += PAGE_SIZE) {
        struct page *pg = virt_to_page(w->kva[off >> PAGE_SHIFT]);
        VM_WARN_ON_ONCE_PAGE(page_ref_count(pg), pg);
        set_page_count(pg, 1);
        rc = vm_insert_page(vma, vma->vm_start + off, pg);
        if (rc) {
            // as the kernel: reset the count (no put_page, so no folio_free path) and free this page and the rest
            // directly; the pages inserted before it are released when the failed mmap's vma is torn down
            set_page_count(pg, 0);
            for (; off < len; off += PAGE_SIZE) pci_free_p2pmem(gpu, w->kva[off >> PAGE_SHIFT], PAGE_SIZE);
            break;
        }
        put_page(pg);
    }
    w->mapped = true;
    return rc;
}

static long xp_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
    return cmd == XOLP2P_MAP ? xp_map_ioctl(f, arg) : -ENOTTY;
}

// misc_open() sets file->private_data to the miscdevice; this driver keeps its per-fd window there, so it starts NULL
static int xp_open(struct inode *ino, struct file *f) { f->private_data = NULL; return 0; }

static int xp_release(struct inode *ino, struct file *f)
{
    mutex_lock(&xp_lock); win_free(f->private_data); f->private_data = NULL; mutex_unlock(&xp_lock);
    return 0;
}

static const struct file_operations fops = { .owner = THIS_MODULE, .open = xp_open, .unlocked_ioctl = xp_ioctl, .mmap = xp_mmap, .release = xp_release };
static struct miscdevice dev = { .minor = MISC_DYNAMIC_MINOR, .name = "xolp2p", .fops = &fops, .mode = 0600 };   // root only
static int __init xp_init(void) { return misc_register(&dev); }
static void __exit xp_exit(void) { misc_deregister(&dev); if (gpu) pci_dev_put(gpu); }
module_init(xp_init);
module_exit(xp_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Xolotl NVMe -> VRAM window (PCI P2PDMA over a pinned BAR1 range)");
