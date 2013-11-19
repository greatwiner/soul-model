/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram neural network ranking language model
 *******************************************************************/
class NgramRankModel : public NeuralModel
{
public:
  //init
  NgramRankModel();
  ~NgramRankModel();
  void
  allocation();

  NgramRankModel(string name, int ngramType, int inputSize, int outputSize,
      int blockSize, int n, int projectionDimension, string nonLinearType,
      intTensor& hiddenLayerSizeArray); // for test only

  NgramRankModel(string name, int ngramType, char* inputVocFileName,
      char* outputVocFileName, int mapCUnk, int mapPUnk, int BOS,
      int blockSize, int n, int projectionDimension, string nonLinearType,
      intTensor& hiddenLayerSizeArray);

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

  void
  trainOne(intTensor& ngram, float learningRate, int subBlockSize);

};

