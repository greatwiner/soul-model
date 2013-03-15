/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for function neural network models
 Input is a vector of float features, output is a class
 *******************************************************************/
class FunctionModel
{
public:
  //init
  FunctionModel();
  ~FunctionModel();
  string name;
  int dim;
  int classNumber;
  int blockSize; // for the bunch mode
  int hiddenLayerSize; // size of hidden layer
  outils* otl;
  FunctionDataSet* dataSet;

  FunctionSequential* baseNetwork;
  LinearSoftmax* outputNetwork;
  floatTensor contextFeature;
  string nonLinearType;
  intTensor hiddenLayerSizeArray;
  int hiddenStep;
  int hiddenNumber;
  floatTensor data;
  floatTensor gradInput;
  intTensor index;
  void
  allocation();

  FunctionModel(int dim, int classNumber, int blockSize, string nonLinearType,
      intTensor& hiddenLayerSizeArray); // for test only

  void
  trainOne(floatTensor& readData, float learningRate, int subBlockSize);

  int
  train(char* dataFileName, int maxExampleNumber, int iteration,
      string learningRateType, float learningRate, float learningRateDecay);
  int
  forward(floatTensor& dataTensor, floatTensor& probTensor);

  int
  computeForward(char* textFileName, int type);

  void
  setWeightDecay(float weightDecay);
  void
  changeBlockSize(int blockSize);

  int
  sequenceTrain(char* prefixModel, int gz, char* prefixData,
      int maxExampleNumber, char* validationFileName, string learningRateType,
      int minIteration, int maxIteration);

  //IO functions
  void
  read(ioFile* iof, int allocation, int blockSize);

  void
  write(ioFile* iof);

};

