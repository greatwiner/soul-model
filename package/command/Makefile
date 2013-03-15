dlvl = ../../.
include $(dlvl)/Makefile.in
all: $(CBIN_PATH)createPrototype.exe $(CBIN_PATH)resamplingData.exe $(CBIN_PATH)sequenceTrain.exe $(CBIN_PATH)prediction.exe $(CBIN_PATH)ovn2ovnb.exe $(CBIN_PATH)text2Perplexity.exe $(CBIN_PATH)text2Prob.exe $(CBIN_PATH)infoModel.exe $(CBIN_PATH)infoNgram.exe $(CBIN_PATH)text2Index.exe $(CBIN_PATH)mapVoc.exe $(CBIN_PATH)takeBaseParameter.exe $(CBIN_PATH)takeLookupTable.exe $(CBIN_PATH)takeParameter.exe $(CBIN_PATH)pushAllParameter.exe $(CBIN_PATH)growPredictionSpace.exe $(CBIN_PATH)growOutPredictionSpace.exe $(CBIN_PATH)resamplingIdData.exe
	
$(CBIN_PATH)createPrototype.exe: createPrototype.cc
	$(CC) $(FLAGS) createPrototype.cc -o $(CBIN_PATH)createPrototype.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)resamplingData.exe: resamplingData.cc
	$(CC) $(FLAGS) resamplingData.cc -o $(CBIN_PATH)resamplingData.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)sequenceTrain.exe: sequenceTrain.cc
	$(CC) $(FLAGS) sequenceTrain.cc -o $(CBIN_PATH)sequenceTrain.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)prediction.exe: prediction.cc
	$(CC) $(FLAGS) prediction.cc -o $(CBIN_PATH)prediction.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)ovn2ovnb.exe: ovn2ovnb.cc
	$(CC) $(FLAGS) ovn2ovnb.cc -o $(CBIN_PATH)ovn2ovnb.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)text2Perplexity.exe: text2Perplexity.cc
	$(CC) $(FLAGS) text2Perplexity.cc -o $(CBIN_PATH)text2Perplexity.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)text2Prob.exe: text2Prob.cc
	$(CC) $(FLAGS) text2Prob.cc -o $(CBIN_PATH)text2Prob.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)infoModel.exe: infoModel.cc
	$(CC) $(FLAGS) infoModel.cc -o $(CBIN_PATH)infoModel.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)infoNgram.exe: infoNgram.cc
	$(CC) $(FLAGS) infoNgram.cc -o $(CBIN_PATH)infoNgram.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)text2Index.exe: text2Index.cc
	$(CC) $(FLAGS) text2Index.cc -o $(CBIN_PATH)text2Index.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)mapVoc.exe: mapVoc.cc
	$(CC) $(FLAGS) mapVoc.cc -o $(CBIN_PATH)mapVoc.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)takeBaseParameter.exe: takeBaseParameter.cc
	$(CC) $(FLAGS) takeBaseParameter.cc -o $(CBIN_PATH)takeBaseParameter.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)takeLookupTable.exe: takeLookupTable.cc
	$(CC) $(FLAGS) takeLookupTable.cc -o $(CBIN_PATH)takeLookupTable.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)takeParameter.exe: takeParameter.cc
	$(CC) $(FLAGS) takeParameter.cc -o $(CBIN_PATH)takeParameter.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)pushAllParameter.exe: pushAllParameter.cc
	$(CC) $(FLAGS) pushAllParameter.cc -o $(CBIN_PATH)pushAllParameter.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)growPredictionSpace.exe: growPredictionSpace.cc
	$(CC) $(FLAGS) growPredictionSpace.cc -o $(CBIN_PATH)growPredictionSpace.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)growOutPredictionSpace.exe: growOutPredictionSpace.cc
	$(CC) $(FLAGS) growOutPredictionSpace.cc -o $(CBIN_PATH)growOutPredictionSpace.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)resamplingIdData.exe: resamplingIdData.cc
	$(CC) $(FLAGS) resamplingIdData.cc -o $(CBIN_PATH)resamplingIdData.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)generateText.exe: generateText.cc
	$(CC) $(FLAGS) generateText.cc -o $(CBIN_PATH)generateText.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
	
$(CBIN_PATH)generateText2.exe: generateText2.cc
	$(CC) $(FLAGS) generateText2.cc -o $(CBIN_PATH)generateText2.exe $(CLIB_INCLUDE) -lmodel -lmodule -ltext -ltensor -lioFile $(LLIB)
