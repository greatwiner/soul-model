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
      string learningRateType, float learningRate, float learningRateDecay);
  int
  forwardProbability(intTensor& ngramTensor, floatTensor& probTensor);

  void
  forwardBackwardAllData(char* dataFileString, int maxExampleNumber, int iteration, int* numberExamples,
  		float learningRate, int* accept);

  void
  forwardProbabilityAllData(char* dataFileString, int maxExampleNumber, int iteration);

  void
  updateAllParameters(float learningRate);

  void
  reUpdateParameters(int accept);

  void
  updateAllRandomness(float learningRate);

  void
  resetGradients();

  void
  initializeP();

  void
  trainOne(intTensor& context, intTensor& word, float learningRate, int subBlockSize);

  void
  forwardBackwardOne(intTensor& context, intTensor& word, int subBlockSize);

  float
  calculeH();

};

