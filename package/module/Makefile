dlvl = ../../.
include $(dlvl)/Makefile.in
alev = *.o

all:
	cp *.H $(CINCLUDE_PATH)
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Module.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Linear.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) MaxLinear.cc	
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) BLinear.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) RLinear.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) RRLinear.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) LookupTable.cc		
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Tanh.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Sigmoid.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Sequential.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) LinearSoftmax.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) FunctionSequential.cc		
	$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)/libmodule.a $(alev)


