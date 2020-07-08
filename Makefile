
OUTNAME_RELEASE = sample_cifar
OUTNAME_DEBUG   = sample_cifar_debug
EXTRA_DIRECTORIES = ../common
.NOTPARALLEL:
MAKEFILE ?= ../Makefile.config
include $(MAKEFILE)