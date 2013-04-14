dlvl = ../../.
include $(dlvl)/Makefile.in
alev = $(CLIB_PATH)*.o
all: copy $(CLIB_PATH)libioFile.a

copy:
		cp *.H $(CINCLUDE_PATH)

$(CLIB_PATH)libioFile.a: $(CLIB_PATH)ioFile.o
		$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)libioFile.a $(alev)
		
$(CLIB_PATH)ioFile.o: ioFile.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) ioFile.cc -o $(CLIB_PATH)ioFile.o