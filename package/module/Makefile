dlvl = ../../.
include $(dlvl)/Makefile.in
alev = $(CLIB_PATH)*.o

all: $(CLIB_PATH)libmodule.a

$(CLIB_PATH)libmodule.a: $(CLIB_PATH)Module.o $(CLIB_PATH)Linear.o $(CLIB_PATH)Linear_Bayes.o $(CLIB_PATH)MaxLinear.o $(CLIB_PATH)BLinear.o $(CLIB_PATH)RLinear.o $(CLIB_PATH)RRLinear.o $(CLIB_PATH)LookupTable.o $(CLIB_PATH)LookupTable_Bayes.o $(CLIB_PATH)Tanh.o $(CLIB_PATH)Sigmoid.o $(CLIB_PATH)Sequential.o $(CLIB_PATH)Sequential_Bayes.o $(CLIB_PATH)LinearSoftmax.o $(CLIB_PATH)LinearSoftmax_Bayes.o $(CLIB_PATH)FunctionSequential.o
		cp *.H $(CINCLUDE_PATH)
		$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)libmodule.a $(alev)
	
$(CLIB_PATH)Module.o: Module.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Module.cc -o $(CLIB_PATH)Module.o
		
$(CLIB_PATH)Linear.o: Linear.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Linear.cc -o $(CLIB_PATH)Linear.o
		
$(CLIB_PATH)Linear_Bayes.o: Linear_Bayes.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Linear_Bayes.cc -o $(CLIB_PATH)Linear_Bayes.o
		
$(CLIB_PATH)MaxLinear.o: MaxLinear.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) MaxLinear.cc -o $(CLIB_PATH)MaxLinear.o
		
$(CLIB_PATH)BLinear.o: BLinear.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) BLinear.cc -o $(CLIB_PATH)BLinear.o
		
$(CLIB_PATH)RLinear.o: RLinear.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) RLinear.cc -o $(CLIB_PATH)RLinear.o
		
$(CLIB_PATH)RRLinear.o: RRLinear.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) RRLinear.cc -o $(CLIB_PATH)RRLinear.o
		
$(CLIB_PATH)LookupTable.o: LookupTable.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) LookupTable.cc -o $(CLIB_PATH)LookupTable.o
		
$(CLIB_PATH)LookupTable_Bayes.o: LookupTable_Bayes.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) LookupTable_Bayes.cc -o $(CLIB_PATH)LookupTable_Bayes.o
		
$(CLIB_PATH)Tanh.o: Tanh.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Tanh.cc -o $(CLIB_PATH)Tanh.o
		
$(CLIB_PATH)Sigmoid.o: Sigmoid.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Sigmoid.cc -o $(CLIB_PATH)Sigmoid.o
		
$(CLIB_PATH)Sequential.o: Sequential.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Sequential.cc -o $(CLIB_PATH)Sequential.o
		
$(CLIB_PATH)Sequential_Bayes.o: Sequential_Bayes.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Sequential_Bayes.cc -o $(CLIB_PATH)Sequential_Bayes.o
		
$(CLIB_PATH)LinearSoftmax.o: LinearSoftmax.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) LinearSoftmax.cc -o $(CLIB_PATH)LinearSoftmax.o
		
$(CLIB_PATH)LinearSoftmax_Bayes.o: LinearSoftmax_Bayes.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) LinearSoftmax_Bayes.cc -o $(CLIB_PATH)LinearSoftmax_Bayes.o
		
$(CLIB_PATH)FunctionSequential.o: FunctionSequential.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) FunctionSequential.cc -o $(CLIB_PATH)FunctionSequential.o