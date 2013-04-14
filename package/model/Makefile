dlvl = ../../.
include $(dlvl)/Makefile.in
alev = $(CLIB_PATH)*.o
all: copy $(CLIB_PATH)libmodel.a

copy:
		cp *.H $(CINCLUDE_PATH)

$(CLIB_PATH)libmodel.a: $(CLIB_PATH)NeuralModel.o $(CLIB_PATH)NgramModel.o $(CLIB_PATH)NgramModel_Bayes.o $(CLIB_PATH)NgramRankModel.o $(CLIB_PATH)RecurrentModel.o $(CLIB_PATH)NgramWordTranslationModel.o $(CLIB_PATH)NgramPhraseTranslationModel.o $(CLIB_PATH)FunctionModel.o
		$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)libmodel.a $(alev)
	
$(CLIB_PATH)NeuralModel.o: NeuralModel.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NeuralModel.cc -o $(CLIB_PATH)NeuralModel.o
		
$(CLIB_PATH)NgramModel.o: NgramModel.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramModel.cc -o $(CLIB_PATH)NgramModel.o
		
$(CLIB_PATH)NgramModel_Bayes.o: NgramModel_Bayes.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramModel_Bayes.cc -o $(CLIB_PATH)NgramModel_Bayes.o
		
$(CLIB_PATH)NgramRankModel.o: NgramRankModel.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramRankModel.cc -o $(CLIB_PATH)NgramRankModel.o
		
$(CLIB_PATH)RecurrentModel.o: RecurrentModel.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) RecurrentModel.cc -o $(CLIB_PATH)RecurrentModel.o
		
$(CLIB_PATH)NgramWordTranslationModel.o: NgramWordTranslationModel.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramWordTranslationModel.cc -o $(CLIB_PATH)NgramWordTranslationModel.o
		
$(CLIB_PATH)NgramPhraseTranslationModel.o: NgramPhraseTranslationModel.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramPhraseTranslationModel.cc -o $(CLIB_PATH)NgramPhraseTranslationModel.o
		
$(CLIB_PATH)FunctionModel.o: FunctionModel.cc
		$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) FunctionModel.cc -o $(CLIB_PATH)FunctionModel.o


