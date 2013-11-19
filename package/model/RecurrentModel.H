/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for recurrent neural network language model
 Maybe we have bugs here :(
 *******************************************************************/
class RecurrentModel : public NeuralModel
{
public:
  //init
  RecurrentModel();
  ~RecurrentModel();
  void
  allocation();

  RecurrentModel(string name, int inputVocSize, int outputVocSize,
      int blockSize, int n, int projectionDimension, string nonLinearType,
      intTensor& hiddenLayerSizeArray, intTensor& codeWord,
      intTensor& outputNetworkSize); // for test only

  RecurrentModel(string name, char* inputVocFileName, char* outputVocFileName,
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

  float
  distance2(RecurrentModel& anotherModel);

  //IO functions
  void
  read(ioFile* iof, int allocation, int blockSize);
  void
  write(ioFile* iof, int closeFile);
  int m;
};

