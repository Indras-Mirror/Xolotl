// p2pgate.c - P2P probes for the RTX 4090 (2026-11-07)
//
// ioctl 1: resolve a registered VRAM window through nvidia_p2p_get_pages()
//          and report the BAR1 addresses it yields.
// ioctl 2: resolve once, map those pages for a peer PCIe device
//          (nvidia_p2p_dma_map_pages), and while the page table is still held,
//          write a pattern into BAR1 at the window's own address and read it
//          back. Everything happens inside one get/put cycle.
//
// Why one cycle: CliDelThirdPartyP2PVidmemInfoPersistent() frees the duped
// memory handle but leaves the internal VidmemInfo linked, so repeated
// get/put cycles on the same window leave stale state and later resolves
// return NV_ERR_OBJECT_NOT_FOUND. Open the window once and keep it.
//
// The window must already be registered with NV50_THIRD_PARTY_P2P by the
// calling process (see rmreg.c); registration is matched by PID.
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/pci.h>
#include <linux/uaccess.h>
#include "nv-p2p.h"

struct p2pgate_req {
    __u64 va, len;
    __u32 mode;
    __s32 status;
    __u32 page_size, entries;
    __u64 pa0, pa_last;
    __u32 contiguous;
};
#define P2PGATE_IOC _IOWR('p', 1, struct p2pgate_req)

struct p2pgate_probe_req {
    __u64 key, len;             // [in]  registered window
    __u32 peer_bus, peer_slot, peer_func;   // [in]  device that DMAs (the NVMe)
    __u32 gpu_bus, gpu_slot, gpu_func;      // [in]  device owning BAR1
    __u32 test_len;             // [in]  bytes to pattern-test (<= 1 MB)

    __s32 status_get;           // errors, resolve
    __s32 status_map;           //          dma map
    __s32 status_probe;         //          BAR1 write/read

    __u32 entries, page_size;
    __u64 pa0, pa_last;         // BAR1 addresses of the first/last page
    __u64 iova0, iova_last;     // what the peer would use
    __u32 iova_contiguous, identity;

    __u64 base, offset;         // BAR1 resource base / offset tested
    __u64 first, last;          // values read back
    __u32 words, mismatches;
};
#define P2PGATE_IOC_PROBE _IOWR('p', 2, struct p2pgate_probe_req)

/* upper bound on the pattern-test size; enough to prove the aperture without
 * writing megabytes into the framebuffer on every run */
#define P2PGATE_MAX_TEST (16u << 20)

static void free_cb(void *data) { pr_info("p2pgate: free callback (allocation went away)\n"); }

static u64 p2p_page_bytes(u32 page_size_type)
{
    switch (page_size_type) {
    case NVIDIA_P2P_PAGE_SIZE_4KB:   return 0x1000;
    case NVIDIA_P2P_PAGE_SIZE_64KB:  return 0x10000;
    case NVIDIA_P2P_PAGE_SIZE_128KB: return 0x20000;
    }
    return 0;
}

static u32 p2p_contiguous(struct nvidia_p2p_page_table *pt, u32 psz)
{
    u32 i;
    for (i = 1; i < pt->entries; ++i)
        if (pt->pages[i]->physical_address != pt->pages[i - 1]->physical_address + psz)
            return 0;
    return 1;
}

// Write a pattern into BAR1 at bar1_addr and read it back. The caller must hold
// the P2P page table, otherwise the BAR1 mapping does not exist at all.
static int bar1_pattern_probe(struct pci_dev *gpu, u64 bar1_addr, u32 len,
                              u64 *base_out, u64 *off_out, u64 *first_out,
                              u64 *last_out, u32 *words_out, u32 *mism_out)
{
    void __iomem *io;
    u64 base, reslen, off;
    u32 i, words, mism = 0;

    base = pci_resource_start(gpu, 1);
    reslen = pci_resource_len(gpu, 1);
    *base_out = base;
    if (!base || !reslen)
        return -ENODEV;

    if (bar1_addr < base || bar1_addr + len > base + reslen) {
        pr_info("p2pgate: window 0x%llx+0x%x outside BAR1 0x%llx..0x%llx\n",
                (unsigned long long)bar1_addr, len, (unsigned long long)base,
                (unsigned long long)(base + reslen));
        return -ERANGE;
    }
    off = bar1_addr - base;
    *off_out = off;

    io = pci_iomap(gpu, 1, 0);
    if (!io)
        return -ENOMEM;

    words = len / 4;
    for (i = 0; i < words; ++i)
        iowrite32(0xC0DE0000u ^ (i * 0x01010101u), io + off + (i * 4));
    wmb();
    for (i = 0; i < words; ++i)
        if (ioread32(io + off + (i * 4)) != (0xC0DE0000u ^ (i * 0x01010101u)))
            mism++;

    *first_out = ioread32(io + off);
    *last_out = ioread32(io + off + ((words - 1) * 4));
    *words_out = words;
    *mism_out = mism;

    pci_iounmap(gpu, io);

    pr_info("p2pgate: bar1 base 0x%llx off 0x%llx len 0x%x: %u/%u words matched, first 0x%llx last 0x%llx\n",
            (unsigned long long)base, (unsigned long long)off, len,
            words - mism, words, (unsigned long long)*first_out,
            (unsigned long long)*last_out);

    return mism ? -EIO : 0;
}

static long p2pgate_resolve_ioctl(unsigned int cmd, unsigned long arg)
{
    struct p2pgate_req r;
    struct nvidia_p2p_page_table *pt = NULL;
    u32 psz;

    if (cmd != P2PGATE_IOC) return -ENOTTY;
    if (copy_from_user(&r, (void __user *) arg, sizeof(r))) return -EFAULT;

    r.status = r.mode ? nvidia_p2p_get_pages_persistent(r.va, r.len, &pt, 0)
                      : nvidia_p2p_get_pages(0, 0, r.va, r.len, &pt, free_cb, NULL);
    r.page_size = 0; r.entries = 0; r.pa0 = 0; r.pa_last = 0; r.contiguous = 0;

    if (r.status == 0 && pt) {
        r.page_size = pt->page_size;
        r.entries = pt->entries;
        if (pt->entries) {
            r.pa0 = pt->pages[0]->physical_address;
            r.pa_last = pt->pages[pt->entries - 1]->physical_address;
        }
        psz = p2p_page_bytes(pt->page_size);
        r.contiguous = p2p_contiguous(pt, psz);
        pr_info("p2pgate: mode %u OK va 0x%llx len 0x%llx: %u pages (size enum %u), pa0 0x%llx pa_last 0x%llx contiguous %u\n",
                r.mode, (unsigned long long)r.va, (unsigned long long)r.len,
                r.entries, r.page_size, (unsigned long long)r.pa0,
                (unsigned long long)r.pa_last, r.contiguous);
        if (r.mode) nvidia_p2p_put_pages_persistent(r.va, pt, 0);
        else        nvidia_p2p_put_pages(0, 0, r.va, pt);
    } else {
        pr_info("p2pgate: mode %u FAILED %d (va 0x%llx len 0x%llx)\n",
                r.mode, r.status, (unsigned long long)r.va,
                (unsigned long long)r.len);
    }
    return copy_to_user((void __user *) arg, &r, sizeof(r)) ? -EFAULT : 0;
}

static long p2pgate_probe_ioctl(unsigned int cmd, unsigned long arg)
{
    struct p2pgate_probe_req r;
    struct nvidia_p2p_page_table *pt = NULL;
    struct nvidia_p2p_dma_mapping *dm = NULL;
    struct pci_dev *peer = NULL, *gpu = NULL;
    u32 i, psz, len;

    if (cmd != P2PGATE_IOC_PROBE) return -ENOTTY;
    if (copy_from_user(&r, (void __user *) arg, sizeof(r))) return -EFAULT;

    r.status_get = 0; r.status_map = 0; r.status_probe = 0;
    r.entries = 0; r.page_size = 0; r.pa0 = 0; r.pa_last = 0;
    r.iova0 = 0; r.iova_last = 0; r.iova_contiguous = 0; r.identity = 0;
    r.base = 0; r.offset = 0; r.first = 0; r.last = 0;
    r.words = 0; r.mismatches = 0;

    if (!r.len || (r.len & 0xFFFF)) {   /* nvidia_p2p works in 64K units */
        r.status_get = -EINVAL;
        pr_info("p2pgate: bad window len 0x%llx (must be non-zero, 64K aligned)\n",
                (unsigned long long)r.len);
        goto out;
    }
    len = r.test_len ? r.test_len : (4u << 20);
    if (len > P2PGATE_MAX_TEST || (len & 3)) {
        r.status_get = -EINVAL;
        pr_info("p2pgate: bad test_len 0x%x (must be <= 0x%x and 4-byte aligned)\n",
                r.test_len, P2PGATE_MAX_TEST);
        goto out;
    }
    if (len > r.len) len = r.len & ~3u;

    peer = pci_get_domain_bus_and_slot(0, r.peer_bus,
                                       PCI_DEVFN(r.peer_slot, r.peer_func));
    gpu  = pci_get_domain_bus_and_slot(0, r.gpu_bus,
                                       PCI_DEVFN(r.gpu_slot, r.gpu_func));
    if (!peer || !gpu) {
        r.status_map = -ENODEV;
        pr_info("p2pgate: bad pci address peer=%p gpu=%p\n", peer, gpu);
        goto out;
    }
    pr_info("p2pgate: peer %s, bar1 owner %s\n", pci_name(peer), pci_name(gpu));

    /* one resolve; everything else happens while it is held */
    r.status_get = nvidia_p2p_get_pages_persistent(r.key, r.len, &pt, 0);
    if (r.status_get != 0 || pt == NULL) {
        pr_info("p2pgate: get_pages FAILED %d\n", r.status_get);
        goto out;
    }

    r.page_size = pt->page_size;
    r.entries = pt->entries;
    if (!pt->entries) { r.status_map = -ENODATA; goto put; }
    r.pa0 = pt->pages[0]->physical_address;
    r.pa_last = pt->pages[pt->entries - 1]->physical_address;

    r.status_map = nvidia_p2p_dma_map_pages(peer, pt, &dm);
    if (r.status_map != 0 || dm == NULL) {
        pr_info("p2pgate: dma_map_pages FAILED %d (%s)\n",
                r.status_map, pci_name(peer));
        goto probe;
    }
    if (dm->entries) {
        r.iova0 = dm->dma_addresses[0];
        r.iova_last = dm->dma_addresses[dm->entries - 1];
        psz = p2p_page_bytes(dm->page_size_type);
        r.iova_contiguous = 1;
        for (i = 1; i < dm->entries; ++i)
            if (dm->dma_addresses[i] != dm->dma_addresses[i - 1] + psz) {
                r.iova_contiguous = 0;
                break;
            }
        r.identity = (r.iova0 == r.pa0 && r.iova_last == r.pa_last);
    }
    pr_info("p2pgate: dma_map OK %s: %u entries, BAR1 0x%llx..0x%llx -> IOVA 0x%llx..0x%llx (contig %u, identity %u)\n",
            pci_name(peer), dm->entries,
            (unsigned long long)r.pa0, (unsigned long long)r.pa_last,
            (unsigned long long)r.iova0, (unsigned long long)r.iova_last,
            r.iova_contiguous, r.identity);

probe:
    /* the BAR1 mapping only exists while pt is held, so test it now */
    r.status_probe = bar1_pattern_probe(gpu, r.pa0, len, &r.base, &r.offset,
                                        &r.first, &r.last, &r.words,
                                        &r.mismatches);

    if (dm) {
        nvidia_p2p_dma_unmap_pages(peer, pt, dm);
        dm = NULL;
    }
put:
    nvidia_p2p_put_pages_persistent(r.key, pt, 0);
out:
    if (peer) pci_dev_put(peer);
    if (gpu)  pci_dev_put(gpu);
    return copy_to_user((void __user *) arg, &r, sizeof(r)) ? -EFAULT : 0;
}

/* ioctl 3 (Step C): HOLD a registered window - persistent get_pages, kept pinned until the fd closes, so a second
 * module (xolp2p, GPL, no nvidia symbols) can hand the BAR1 range to the kernel P2PDMA layer while the NVMe reads
 * into it.  One held window per open fd. */
struct p2pgate_hold_req {
    __u64 key, len;             // [in]  registered window
    __s32 status;               // [out] nvidia_p2p_get_pages_persistent
    __u32 entries, page_size, contiguous;
    __u64 pa0, pa_last;         // BAR1 bus addresses of the first/last page
};
#define P2PGATE_IOC_HOLD _IOWR('p', 3, struct p2pgate_hold_req)

#define PG_MAGIC 0x50324748u   /* "P2GH": release only ever puts a window HOLD created */
struct p2pgate_held { u32 magic; struct nvidia_p2p_page_table *pt; __u64 key; };

static long p2pgate_hold_ioctl(struct file *f, unsigned long arg)
{
    struct p2pgate_hold_req r;
    struct p2pgate_held *h = f->private_data;
    struct nvidia_p2p_page_table *pt = NULL;
    u32 i;
    if (h) return -EBUSY;   // private_data is NULL from p2pgate_open, so only a second HOLD on this fd lands here
    if (copy_from_user(&r, (void __user *) arg, sizeof(r))) return -EFAULT;
    r.status = nvidia_p2p_get_pages_persistent(r.key, r.len, &pt, 0);
    r.entries = r.page_size = r.contiguous = 0; r.pa0 = r.pa_last = 0;
    if (r.status == 0 && pt && pt->entries) {
        r.entries = pt->entries; r.page_size = pt->page_size;
        r.pa0 = pt->pages[0]->physical_address; r.pa_last = pt->pages[pt->entries - 1]->physical_address;
        r.contiguous = 1;
        for (i = 1; i < pt->entries; ++i)
            if (pt->pages[i]->physical_address != pt->pages[i - 1]->physical_address + 0x10000) { r.contiguous = 0; break; }
        h = kzalloc(sizeof(*h), GFP_KERNEL);
        if (!h) { nvidia_p2p_put_pages_persistent(r.key, pt, 0); return -ENOMEM; }
        h->magic = PG_MAGIC; h->pt = pt; h->key = r.key;
        f->private_data = h;
        pr_info("p2pgate: HOLD key 0x%llx len 0x%llx -> %u pages, BAR1 0x%llx..0x%llx contiguous %u\n",
                r.key, r.len, r.entries, r.pa0, r.pa_last, r.contiguous);
    } else {
        pr_info("p2pgate: HOLD FAILED %d (key 0x%llx len 0x%llx)\n", r.status, r.key, r.len);
    }
    return copy_to_user((void __user *) arg, &r, sizeof(r)) ? -EFAULT : 0;
}

// misc_open() sets file->private_data to the miscdevice; HOLD keeps its per-fd window there, so it starts NULL
static int p2pgate_open(struct inode *ino, struct file *f) { f->private_data = NULL; return 0; }

static int p2pgate_release(struct inode *ino, struct file *f)
{
    struct p2pgate_held *h = f->private_data;
    if (h && h->magic != PG_MAGIC) { WARN_ONCE(1, "p2pgate: refusing to release a non-HOLD pointer\n"); return 0; }
    if (h) {
        h->magic = 0;
        nvidia_p2p_put_pages_persistent(h->key, h->pt, 0);
        pr_info("p2pgate: released held window key 0x%llx\n", h->key);
        kfree(h);
        f->private_data = NULL;
    }
    return 0;
}

static long p2pgate_dispatch(struct file *f, unsigned int cmd, unsigned long arg)
{
    if (_IOC_TYPE(cmd) == 'p') {
        switch (_IOC_NR(cmd)) {
        case 1: return p2pgate_resolve_ioctl(cmd, arg);
        case 2: return p2pgate_probe_ioctl(cmd, arg);
        case 3: return p2pgate_hold_ioctl(f, arg);
        }
    }
    return -ENOTTY;
}

static const struct file_operations fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = p2pgate_dispatch,
    .compat_ioctl = p2pgate_dispatch,
    .open = p2pgate_open,
    .release = p2pgate_release,
};

static struct miscdevice dev = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "p2pgate",
    .fops = &fops,
    .mode = 0600,   /* root only */
};

static int __init p2pgate_init(void)
{
    pr_info("p2pgate: loaded, resolve %zu bytes, probe %zu bytes\n",
            sizeof(struct p2pgate_req), sizeof(struct p2pgate_probe_req));
    return misc_register(&dev);
}

static void __exit p2pgate_exit(void) { misc_deregister(&dev); }

module_init(p2pgate_init);
module_exit(p2pgate_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("GeForce GPUDirect P2P probes (resolve, peer map, BAR1 window test)");
