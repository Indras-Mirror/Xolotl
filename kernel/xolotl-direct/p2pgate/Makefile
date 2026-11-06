# p2pgate: pins a VRAM window through the NVIDIA driver's P2P API (nvidia_p2p_get_pages_persistent).
# Needs the NVIDIA module's Module.symvers for those symbols: found under /var/lib/dkms (DKMS installs) or set NV_SYMVERS.
obj-m := p2pgate.o
KDIR ?= /lib/modules/$(shell uname -r)/build
NV_SYMVERS ?= $(firstword $(wildcard /var/lib/dkms/nvidia/*/$(shell uname -r)/*/module/Module.symvers) \
                         $(wildcard /usr/src/nvidia-*/Module.symvers))
KBUILD_EXTRA_SYMBOLS := $(NV_SYMVERS)
export KBUILD_EXTRA_SYMBOLS
all:
	@test -n "$(NV_SYMVERS)" || { echo "p2pgate: NVIDIA Module.symvers not found - make NV_SYMVERS=/path/to/Module.symvers"; exit 1; }
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules
clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
