dlvl = ../../.
include $(dlvl)/Makefile.in
all:
	$(CC) $(FLAGS) test.cc -o $(CBIN_PATH)test.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	$(CC) $(FLAGS) forwardTrainTest.cc -o $(CBIN_PATH)forwardTrainTest.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)