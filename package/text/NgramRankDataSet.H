/*******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram ranking data set:
 Separate positive and negative examples. Positive example is read
 from data, negative is created by changing the last, first or center
 (depending on type) word to a word selected randomly with an uniform
 distribution.
 It takes as input all word of an n-gram.
 Output layer has 1 node. For each training example, update parameter
 if score(positive) - score(negative) < 1.00

 int type: defined in DataSet.H
 0: Normal, 1: Inverse, 2: Center
 See DataSet.H for more detail about functions
 *******************************************************************/

class NgramRankDataSet : public DataSet
{
public:
  // From DataSet
  NgramRankDataSet(int type, int n, int BOS, SoulVocab* inputVoc,
      SoulVocab* outputVoc, int mapIUnk, int mapOUnk, int maxNgramNumber);
  int
  setMaxNgramNumber(int maxNgramNumber);

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
  readCoBiNgram(ioFile* iof);

  //Other functions
  void
  shuffle(int times);

  void
  sortNgram();
  string
  inverse(string line);
  int
  writeReBiNgram();
  float
  computePerplexity();

};

