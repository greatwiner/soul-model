dlvl = ../../.
include $(dlvl)/Makefile.in
alev = *.o
all:
	cp *.H $(CINCLUDE_PATH)
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) outils.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Tensor.cc	
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) floatTensor.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) intTensor.cc	
	$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)/libtensor.a $(alev)


