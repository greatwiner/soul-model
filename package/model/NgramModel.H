/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram neural network language model
 *******************************************************************/
class NgramModel : public NeuralModel
{
public:
  NgramModel();
  ~NgramModel();
  // Constructor
  void
  allocation();

  // This constructor is for debugging
  NgramModel(string name, int inputSize, int outputSize, int blockSize, int n,
      int projectionDimension, string nonLinearType,
      intTensor& hiddenLayerSizeArray, intTensor& codeWord,
      intTensor& outputNetworkSize);

  // Constructor, read parameters to create a model:
  // name: name or type of the model
  // inputVocFileName, outputVocFileName: input and output vocabs
  // mapIUnk: map unknown context words to <unk> or not
  // mapOUnk: map unknown predicted words to <unk> or not
  // BOS always = n - 1, use neural networks for all possible n-grams, see DataSet
  // nonLinearType = 's': sigmoid, 't': tangent hyperbolic, 'l': linear
  // hiddenLayerSizeArray: sizes of hidden layers
  // codeWordFileName, outputNetworkSizeFileName: binary files,
  // if codeWordFileName, outputNetworkSizeFileName = xxx, use 1 softmax layer,
  // don't use tree structure
  NgramModel(string name, char* inputVocFileName, char* outputVocFileName,
      int mapIUnk, int mapOUnk, int BOS, int blockSize, int n,
      int projectionDimension, string nonLinearType,
      intTensor& hiddenLayerSizeArray, char* codeWordFileName,
      char* outputNetworkSizeFileName);

  // firstTime don't do anything, only for recurrent models but
  // need to declare here because they are pure virtual functions in
  // the mother class NgramModel
  void
  firstTime();
  void
  firstTime(intTensor& context);

  // Train models, train functions for different types of models
  // (Ngram, Recurrent...) are different mainly in the way of reading n-grams,
  // they all call trainOne function of NeuralModel
  int
  train(char* dataFileName, int maxExampleNumber, int iteration,
      string learningRateType, float learningRate, float learningRateDecay);

  // Compute probabilities for n-gram in ngramTensor, output to probTensor
  int
  forwardProbability(intTensor& ngramTensor, floatTensor& probTensor);

  //IO functions
  void
  read(ioFile* iof, int allocation, int blockSize);

  void
  write(ioFile* iof, int closeFile);

};

