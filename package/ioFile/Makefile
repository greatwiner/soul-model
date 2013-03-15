dlvl = ../../.
include $(dlvl)/Makefile.in
alev = $(CLIB_PATH)*.o
all: $(CLIB_PATH)ioFile.o linker

$(CLIB_PATH)ioFile.o: ioFile.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) ioFile.cc -o $(CLIB_PATH)ioFile.o
		  
linker: 
		cp *.H $(CINCLUDE_PATH)
		$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)libioFile.a $(alev)