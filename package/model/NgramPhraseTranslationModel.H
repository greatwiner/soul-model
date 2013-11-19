/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram neural network phrase factored translation model
 *******************************************************************/
class NgramPhraseTranslationModel : public NeuralModel
{
public:
  //init
  NgramPhraseTranslationModel();
  ~NgramPhraseTranslationModel();

  void
  allocation();
  int nm;
  NgramPhraseTranslationModel(string name, int ngramType, int inputSize, int outputSize,
      int blockSize, int n, int projectionDimension, string nonLinearType,
      intTensor& hiddenLayerSizeArray, intTensor& codeWord,
      intTensor& outputNetworkSize); // for test only

  NgramPhraseTranslationModel(string name, int ngramType, char* inputVocFileName,
      char* outputVocFileName, int mapCUnk, int mapPUnk, int BOS,
      int blockSize, int n, int projectionDimension, string nonLinearType,
      intTensor& hiddenLayerSizeArray, char* codeWordFileName,
      char* outputNetworkSizeFileName);

  void
  firstTime();
  void
  firstTime(intTensor& context);
  int
  train(char* dataFileName, int maxExampleNumber, int iteration,
      string learningRateType, float learningRate, float learningRateDecay);
  int
  forwardProbability(intTensor& ngramTensor, floatTensor& probTensor);

  //IO functions
  void
  read(ioFile* iof, int allocation, int blockSize);
  void
  write(ioFile* iof, int closeFile);

};

