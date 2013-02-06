/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 General class for data set.
 It can read data from text, read; sort and rasampling n-gram,
 read, write n-gram to a file as tensor, output probabilities...

 *******************************************************************/

class DataSet
{
public:
  DataSet();
  virtual
  ~DataSet();

  // Data type, see sub-classes for detail
  int type;
  // Pointer to integer array for all n-grams
  int* data;
  // Maximal ngram number can be handled
  int maxNgramNumber;

  // dataTensor is a tensor that points to n-gram data
  // dataTensor.data = data
  // Each n-gram is composed of (n + 3) int:
  // id_1 id_2 ... id_(n - 1) id_n ID_END_NGRAM idngram info
  // id_1 ... id_n is indices of words.
  // ID_END_NGRAM is used to separate n-gram and other infos
  // mainly for sort function.
  intTensor dataTensor;
  // floatTensor keeps output probabilities for n-gram in dataTensor
  floatTensor probTensor;
  // Number of n-grams
  int ngramNumber;
  // Output perplexity
  float perplexity;
  // Order
  int n;
  // Two parameter for recurrent models, need to know the size of
  // training block; if models consider previous sentences (cont = 1)
  int blockSize;
  int cont;

  // BOS: Begin of sentence, considering n-grams with how many <s> at
  // the beginning. BOS: max number of <s>
  // n = 4, BOS = 2 => Considering <s> <s> x y, <s> x y z, x y z t
  // but don't care <s> <s> <s> x
  // Now always use BOS = n - 1 (process all n-grams possible)
  int BOS;
  // Vocabulary for the input and output sizes, can be different
  SoulVocab* inputVoc;
  SoulVocab* outputVoc;
  // Map the index for unknown words to the index of UNK or not
  int mapIUnk;
  int mapOUnk;
  // Group n-grams sharing the same context or not while forwarding
  // if group, need to sort n-grams
  int groupContext;
  // Change ngramNumber to 0, start to read n-grams from the
  // beginning of data
  void
  reset();
  // If line is all space?
  int
  checkBlankString(string line);

  // Some functions are virtual (see sub-classes for actual implementation)

  // addLine: read one line and stock all satisfying n-grams in data
  // array
  virtual
  int
  addLine(string line) = 0;
  virtual int
  addLine(ioFile* iof) = 0;
  // output the random subset of indices of setences in the training
  // data
  virtual int
  resamplingSentence(int totalLineNumber, int resamplingLineNumber,
      int* resamplingLineId) = 0;

  // Read a file, meaning call addLine with all lines of this file
  virtual int
  readText(ioFile* iof) = 0;

  // Resampling n-gram from files, meaning call addLine with
  // subset of lines of this file
  virtual int
  resamplingText(ioFile* iof, int totalLineNumber, int resamplingLineNumber) = 0;

  // Read file data description to do data resampling
  // Format of dataDesFile:
  // name_of_file number_of_lines resampling_rate
  int
  resamplingDataDes(char* dataDesFileName, int type);

  // Map dataTensor to data, sorting n-gram if model is n-gram
  // and context grouping is set. Create also memory for probTensor.
  // To process with other classes (computing probability)
  // using DataSet.dataTensor
  virtual intTensor&
  createTensor() = 0;

  // From probTensor, computing perplexity (for special models, can be
  // actually a score, not perplexity)
  virtual float
  computePerplexity() = 0;

  // Read n-gram from n-gram text, each line has n words, for example
  // a b c d
  // e f g h
  virtual int
  readTextNgram(ioFile* iof) = 0;

  // Write n-grams to binary file with resampling format (model will use it
  // to learn parameters. It has a header with 2 integers:
  // number_of_examples and order
  virtual void
  writeReBiNgram(ioFile* iof) = 0;

  // Write n-grams to binary file with computing format (model will use it
  // to compute probabilities. It has a header with 1 integer:
  // number_of_examples
  virtual int
  readCoBiNgram(ioFile* iof) = 0;

};
