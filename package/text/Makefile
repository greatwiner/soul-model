dlvl = ../../.
include $(dlvl)/Makefile.in
alev = $(CLIB_PATH)*.o
all: copy $(CLIB_PATH)libtext.a

copy:
		cp *.H $(CINCLUDE_PATH)

$(CLIB_PATH)libtext.a: $(CLIB_PATH)Vocab.o $(CLIB_PATH)DataSet.o $(CLIB_PATH)NgramDataSet.o $(CLIB_PATH)NgramRankDataSet.o $(CLIB_PATH)NgramNCEDataSet.o $(CLIB_PATH)RecurrentDataSet.o $(CLIB_PATH)NgramWordTranslationDataSet.o $(CLIB_PATH)NgramPhraseTranslationDataSet.o $(CLIB_PATH)FunctionDataSet.o
		$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)libtext.a $(alev)
	
$(CLIB_PATH)Vocab.o: Vocab.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) Vocab.cc -o $(CLIB_PATH)Vocab.o
		
$(CLIB_PATH)DataSet.o: DataSet.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) DataSet.cc -o $(CLIB_PATH)DataSet.o
	
$(CLIB_PATH)NgramDataSet.o: NgramDataSet.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramDataSet.cc -o $(CLIB_PATH)NgramDataSet.o
		
$(CLIB_PATH)NgramRankDataSet.o: NgramRankDataSet.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramRankDataSet.cc -o $(CLIB_PATH)NgramRankDataSet.o
		
$(CLIB_PATH)NgramNCEDataSet.o: NgramNCEDataSet.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramNCEDataSet.cc -o $(CLIB_PATH)NgramNCEDataSet.o
		
$(CLIB_PATH)RecurrentDataSet.o: RecurrentDataSet.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) RecurrentDataSet.cc -o $(CLIB_PATH)RecurrentDataSet.o
		
$(CLIB_PATH)NgramWordTranslationDataSet.o: NgramWordTranslationDataSet.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramWordTranslationDataSet.cc -o $(CLIB_PATH)NgramWordTranslationDataSet.o
		
$(CLIB_PATH)NgramPhraseTranslationDataSet.o: NgramPhraseTranslationDataSet.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramPhraseTranslationDataSet.cc -o $(CLIB_PATH)NgramPhraseTranslationDataSet.o
		
$(CLIB_PATH)FunctionDataSet.o: FunctionDataSet.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) FunctionDataSet.cc -o $(CLIB_PATH)FunctionDataSet.o