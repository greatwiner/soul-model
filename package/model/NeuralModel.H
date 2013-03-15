/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Abstract class for neural network language model
 *******************************************************************/
class NeuralModel
{
public:
  string name;
  int recurrent;
  int ngramType;
  int reserve;
  // for the bunch mode
  int blockSize;
  // n of n-gram -> n - 1 words in the history (inputs)
  int n;
  // For bunch mode: a vector which contains
  // all the corresponding outputs.
  floatTensor probabilityOne;
  // the dimension of the projection space
  // for each word of the context.
  int dimensionSize;
  // size of hidden layer
  int hiddenLayerSize;

  outils* otl;
  // Vocabulary
  SoulVocab* inputVoc;
  SoulVocab* outputVoc;
  // Deal with unknown input (output) word or not
  int mapIUnk, mapOUnk;
  // Begin of sentence, add BOS * <s> for each sentence
  // Always = length of ngram - 1 => take all ngram possbile
  int BOS;
  // For recurrent, take context over sentence or only in sentence
  int cont;
  // Probability of unknown word is multiplied by incrUnk
  // Default = 1 means not use.
  // Attention: Input for the commands (text2Prob...) is different,
  // incrUnk of a model = 10^(given incrUnk)
  float incrUnk;
  // Use data set to manipulate data, each type of model has its own
  // type of dataSet
  DataSet* trainingDataSet;
  DataSet* dataSet;

  // For SOUL
  // Sequential of modules, first part of neural network
  // LookupTable -> Linear -> ... -> First hidden layer
  Sequential* baseNetwork;
  // Layers of softmax for output tree
  LinearSoftmax** outputNetwork;

  // Code of words in the tree
  intTensor codeWord;
  // Max length of codes of words (= 2 x depth of the tree)
  int maxCodeWordLength;
  // Size of outputNetwork and its elements, useful to read and write model
  intTensor outputNetworkSize;
  int outputNetworkNumber;
  // Example for codeWord and outputNetworkSize
  // a,b : shortlist words, c,d,e: in one top class, d,e is in one sub-class
  // codeWord:
  // a = [0, 1, -1, -1, -1, -1]
  // b = [0, 2, -1, -1, -1, -1]
  // c = [0, 0,  1,  0, -1, -1]
  // d = [0, 0,  1,  1,  2,  0]
  // e = [0, 0,  1,  1,  2,  1]
  // Number in impair positions: layer index, in pair positions: child number,
  // -1 marks the end of the code
  // maxCodeWordLength = 6
  // outputNetworkSize: [3, 2, 2]
  // outputNetworkNumber = 3

  // Type of activation
  string nonLinearType;
  // Size of each hidden layers
  intTensor hiddenLayerSizeArray;
  // Number of hidden layers
  int hiddenNumber;
  // hiddenStep = 1 if linear activation, = 2 if non linear activation
  // is used
  int hiddenStep;

  // Variables for computing
  // Bits to say which element in outputNetwork has done forwarding,
  // If = 1, don't need to forward it again
  intTensor doneForward;
  // Tensor which is output of main softmax (probability of words
  // in shortlist and main classes of words out of shortlist
  floatTensor mainProb;
  // Output of first hidden layer
  floatTensor contextFeature;
  // Tensor which is the values backward from outputNetwork
  floatTensor gradContextFeature;
  // To process return values
  floatTensor gradInput;
  floatTensor selectContextFeature;
  floatTensor selectGradContextFeature;
  // To process code of words
  intTensor localCodeWord;
  intTensor selectCodeWord;
  intTensor selectLocalCodeWord;

  // Constructor, call this before doing the other construction.
  // Unless, iof can't be used
  NeuralModel();
  virtual
  ~NeuralModel();
  // Build a model with allocation
  virtual void
  allocation() = 0;
  // word is tensor (bs x 1) which represents predicted words
  // decodeWord copy codes of these words into localCodeWord
  // (maxCodeWordLength x bs)
  // For unused n-grams in block, column = SIGN_NOT_WORD
  int
  decodeWord(intTensor& word);
  int
  decodeWord(intTensor& word, int subBlockSize);

  // Edit weights which represent word spaces
  void
  setWeight(char* layerName, floatTensor& tensor);
  floatTensor&
  getWeight(char* layerName);
  void
  setWeightDecay(float weightDecay);

  // Change block size, the number of examples in bunch mode
  void
  changeBlockSize(int blockSize);

  // Compute probabilities,
  // In case of n-gram, context is an intTensor ([n - 1] x bs) which
  // groups the input contexts in bunch mode, each column contains indices
  // of (n - 1) previous words
  // word is the vector (bs x 1) which represents words to be predicted
  // forward and update the probabilities in probabilityOne (bs x 1)
  floatTensor&
  forwardOne(intTensor& context, intTensor& word);

  // Compute probabilities for a text file,
  // textType == "n": 'normal', normal text
  // textType = "l": 'list' of n-grams, each line contains n words
  // textType == "id": 'index' of n-grams, file is in binary format with
  // header = 1 integer, order of n-gram, then each n integers: word indices
  // Read all n-grams in dataTensor of DataSet, compute and write in
  // probTensor of DataSet
  floatTensor&
  computeProbability(DataSet* dataset, char* textFileName, string textType);

  // Compute perplexity, similar to computeProbability but about perplexity
  float
  computePerplexity(DataSet* dataset, char* textFileName, string textType);

  // Compute probabilities, suppose that n-grams are already read in DataSet
  floatTensor&
  computeProbability();

  // Compute perplexity, suppose that n-grams are already read in DataSet
  float
  computePerplexity();

  // Compute probabilites for ngram in ngramTensor, write to probTensor
  virtual int
  forwardProbability(intTensor& ngramTensor, floatTensor& probTensor) = 0;

  // Only for recurrent models
  virtual void
  firstTime() = 0;
  virtual void
  firstTime(intTensor& context) = 0;

  // trainOne: Train with one example or one block of examples
  // shared with all sub-classes
  void
  trainOne(intTensor& context, intTensor& word, float learningRate);
  void
  trainOne(intTensor& context, intTensor& word, float learningRate,
      int subBlockSize);

  // Only for test and debugging with artificial data
  int
  trainTest(int maxExampleNumber, float weightDecay, string learningRateType,
      float learningRate, float learningRateDecay, intTensor& gcontext,
      intTensor& gword);

  // train: Train with one n-gram file (1 epoch)
  virtual int
  train(char* dataFileName, int maxExampleNumber, int iteration,
      string learningRateType, float learningRate, float learningRateDecay) = 0;

  // sequenceTrain with several epochs using early stopping,
  // call train(...) for several times
  // maxExampleNumber = 0 means using all examples in resampling data files
  // learningRateType is
  // 'n': normal: learningRate = function of seen examples
  // 'd': down: learningRate is fixed for each epoch, is divided by learningRateDecay
  // if perplexity of dev data increases
  // if prefixModel = 'xxx': Don't write models to file
  int
  sequenceTrain(char* prefixModel, int gz, char* prefixData,
      int maxExampleNumber, char* trainingFileName, char* validationFileName, string validType,
      string learningRateType, int minIteration, int maxIteration);

  // Read, write with file
  virtual void
  read(ioFile* iof, int allocation, int blockSize) = 0;
  virtual void
  write(ioFile* iof) = 0;
};

