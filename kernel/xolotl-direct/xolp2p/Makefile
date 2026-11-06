# xolp2p: registers the pinned VRAM window with the kernel PCI P2PDMA layer so the NVMe drive can write into it (GPL, no NVIDIA symbols).
obj-m := xolp2p.o
KDIR ?= /lib/modules/$(shell uname -r)/build
all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules
clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
