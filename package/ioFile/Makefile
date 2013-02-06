dlvl = ../../.
include $(dlvl)/Makefile.in
alev = *.o
all:
	cp *.H $(CINCLUDE_PATH)	
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) ioFile.cc
	$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)/libioFile.a $(alev)

