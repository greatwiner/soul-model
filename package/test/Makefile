dlvl = ../../.
include $(dlvl)/Makefile.in
all: $(CBIN_PATH)test.exe $(CBIN_PATH)forwardTrainTest.exe

$(CBIN_PATH)test.exe: test.cc
	$(CC) $(FLAGS) test.cc -o $(CBIN_PATH)test.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)forwardTrainTest.exe: forwardTrainTest.cc
	$(CC) $(FLAGS) forwardTrainTest.cc -o $(CBIN_PATH)forwardTrainTest.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)