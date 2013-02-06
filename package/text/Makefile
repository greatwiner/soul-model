dlvl = ../../.
include $(dlvl)/Makefile.in
alev = *.o
all:
	cp *.H $(CINCLUDE_PATH)
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Vocab.cc		
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) DataSet.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramDataSet.cc	
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramRankDataSet.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) RecurrentDataSet.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramWordTranslationDataSet.cc	
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramPhraseTranslationDataSet.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) FunctionDataSet.cc
	$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)/libtext.a $(alev)
	
