class NgramModel_Bayes : public NgramModel
{
public:

	NgramModel_Bayes();

	NgramModel_Bayes(string name, int inputSize, int outputSize, int blockSize, int n,
		      int projectionDimension, string nonLinearType,
		      intTensor& hiddenLayerSizeArray, intTensor& codeWord,
		      intTensor& outputNetworkSize);
	NgramModel_Bayes(string name, char* inputVocFileName, char* outputVocFileName,
		      int mapIUnk, int mapOUnk, int BOS, int blockSize, int n,
		      int projectionDimension, string nonLinearType,
		      intTensor& hiddenLayerSizeArray, char* codeWordFileName,
		      char* outputNetworkSizeFileName);
	~NgramModel_Bayes();
	float trainLikel;

  void
  firstTime();
  int
  train(char* dataFileName, int maxExampleNumber, int iteration,
      string learningRateType, float learningRateForRd, float learningRateForParas, float learningRateDecay);
  /*int
  forwardProbability(intTensor& ngramTensor, floatTensor& probTensor);*/

  void
  forwardBackwardAllData(char* dataFileString, int maxExampleNumber, int iteration,
  		float learningRateForRd, float learningRateForParas);

  void
  forwardProbabilityAllData(char* dataFileString, int maxExampleNumber, int iteration);

  void
  updateAllParameters(float learningRateForRd, float learningRateForParas, int last);

  void
  updateAllRandomness(float learningRate);

  /*void
  resetGradients();*/

  void
  initializeP();

  void
  trainOne(intTensor& context, intTensor& word, float learningRateForRd, float learningRateForParas, int subBlockSize, int last);

  void
  forwardBackwardOne(intTensor& context, intTensor& word, int subBlockSize, int last, float learningRateForRd, float learningRateForParas);

  float
  calculeH(float RATE);

};

