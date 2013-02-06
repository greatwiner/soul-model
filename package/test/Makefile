dlvl = ../../.
include $(dlvl)/Makefile.in
all:
	$(CC) $(FLAGS) test.cc -o test.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	$(CC) $(FLAGS) forwardTrainTest.cc -o forwardTrainTest.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	cp *.exe $(CBIN_PATH)

