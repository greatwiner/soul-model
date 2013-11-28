/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for data set of n-gram models.
 See DataSet.H for more detail about functions
 int type: defined in DataSet.H
 0: Normal, 1: Inverse, 2: Center
 *******************************************************************/

class NgramDataSet : public DataSet
{
public:
  // From DataSet
  NgramDataSet(int type, int n, int BOS, SoulVocab* inputVoc,
      SoulVocab* outputVoc, int mapIUnk, int mapOUnk, int maxNgramNumber);
  NgramDataSet(int n, int maxNgramNumber);

  int
  addLine(string line);
  int
  addLine(ioFile* iof);
  int
  resamplingSentence(int totalLineNumber, int resamplingLineNumber,
      int* resamplingLineId);
  int
  readText(ioFile* iof);
  int
  resamplingText(ioFile* iof, int totalLineNumber, int resamplingLineNumber);

  intTensor&
  createTensor();

  int
  readTextNgram(ioFile* iof);

  void
  writeReBiNgram(ioFile* iof);

  int
  writeReBiNgram();

  int
  readCoBiNgram(ioFile* iof);

  float
  computePerplexity();

  //Other functions:

  // Shuffle n-gram randomly, to do data resampling
  void
  shuffle(int times);

  // Sort n-gram using quicksort, to do context grouping after
  void
  sortNgram();

  // Create a new line with words from right to left, for inverse model
  string
  inverse(string line);

  // Functions to resample n-grams with files containing indices of words)
  // File header contains one integer: order n
  int
  resamplingIdDataDes(char* dataDesFileName);
  int
  resamplingId(ioFile* iof, int totalLineNumber, int resamplingLineNumber);

};

