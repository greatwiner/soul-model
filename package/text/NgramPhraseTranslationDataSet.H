/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram phrase based data set.
 See DataSet.H and NgramDataSet.{cc,h} for more detail.
 outputVoc needs to be the first part of inputVoc
 There is always one <s>, </s>, <UNK> for two languages.

 int type defined in DataSet.H
 If en -> fr
 0: TrgSrc: p(f_n/e_n f_n-1 e_n-1)
 1: Trg: p(f_n/f_n-1 e_n-1)
 2: SrcTrg: p(e_n/f_n f_n-1 e_n-1)
 3: Src: p(e_n/f_n-1 e_n-1)

 Order (n) is defined as follows:
 TrgSrc predicts a target phrase based on the (n - 1) previous
 target phrases and the n source phrases (with the current source
 phrase). nm = n * 2 is the actually value used
 Trg predicts a target phrase based on the (n - 1) previous
 target phrases and the (n - 1) source phrase (without the current
 source phrase).  nm = n * 2 - 1 is the actually value used
 Same for SrcTrg, Src
 *******************************************************************/
class NgramPhraseTranslationDataSet : public DataSet
{
public:
  int nm;
  //From DataSet
  NgramPhraseTranslationDataSet(int type, int n, int BOS, SoulVocab* inputVoc,
      SoulVocab* outputVoc, int mapIUnk, int mapOUnk, int maxNgramNumber);
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

  // Other functions

  // Function for addLine, verify and add satisfying n-gram
  // depending on type
  int
  addDisTuple(int* srcIndex, int* desIndex, int* unkIndex);

  void
  shuffle(int times);
  void
  sortNgram();
  int
  writeReBiNgram();
  float
  computePerplexity();

};

