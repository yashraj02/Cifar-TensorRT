
OUTNAME_RELEASE = cifar
OUTNAME_DEBUG   = sample_cifar_debug
EXTRA_DIRECTORIES = ../cifar
.NOTPARALLEL:
MAKEFILE ?= ../Makefile.config
include $(MAKEFILE)