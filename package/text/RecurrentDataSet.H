/*******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for recurrent data set.
 See DataSet.H for more detail about functions
 *******************************************************************/

class RecurrentDataSet : public DataSet
{
public:

  // indicates if the data has been organized
  int alreadyCreateTensor;
  // From DataSet
  RecurrentDataSet(int n, SoulVocab* inputVoc, SoulVocab* outputVoc, int cont,
      int blockSize, int maxNgramNumber);

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

  float
  computePerplexity();
  // Other function

  // Find possible position to cut sentence flow into several ones, for
  // bunch mode
  int
  findPos(intTensor& pos, intTensor& dis);

  void
  shuffle(int times);

  int
  writeReBiNgram();

};
