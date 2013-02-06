dlvl = ../../.
include $(dlvl)/Makefile.in
alev = *.o
all:
	cp *.H $(CINCLUDE_PATH)
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NeuralModel.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramModel.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramRankModel.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) RecurrentModel.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramWordTranslationModel.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) NgramPhraseTranslationModel.cc
	$(CC) $(CFLAGS_OBJECT) $(CLIB_INCLUDE) FunctionModel.cc
	$(ARCH) $(ARCHFLAGS)  $(CLIB_PATH)/libmodel.a $(alev)


