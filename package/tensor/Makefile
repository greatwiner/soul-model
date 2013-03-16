dlvl = ../../.
include $(dlvl)/Makefile.in
alev = $(CLIB_PATH)*.o
all: $(CLIB_PATH)libtensor.a

$(CLIB_PATH)libtensor.a: $(CLIB_PATH)outils.o $(CLIB_PATH)Tensor.o $(CLIB_PATH)floatTensor.o $(CLIB_PATH)intTensor.o
		cp *.H $(CINCLUDE_PATH)
		$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)libtensor.a $(alev)

$(CLIB_PATH)outils.o: outils.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) outils.cc -o $(CLIB_PATH)outils.o
		
$(CLIB_PATH)Tensor.o: Tensor.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Tensor.cc -o $(CLIB_PATH)Tensor.o
		
$(CLIB_PATH)floatTensor.o: floatTensor.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) floatTensor.cc -o $(CLIB_PATH)floatTensor.o
		
$(CLIB_PATH)intTensor.o: intTensor.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) intTensor.cc -o $(CLIB_PATH)intTensor.o