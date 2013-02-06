/*******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for data set for function model. It takes float features to
 predict probabilities for classes.
 See DataSet.H and NgramDataSet.H for more detail about functions
 *******************************************************************/

class FunctionDataSet
{
public:
  FunctionDataSet(int dim, int classNumber);
  ~FunctionDataSet();
  // Dimension of float feature vector
  int dim;
  // Number of examples
  int dataNumber;
  // Number of classes
  int classNumber;
  floatTensor dataTensor;
  floatTensor probTensor;
  float* data;
  int
  addLine(string line);
  int
  readText(ioFile* iof);

  void
  createTensor();

  void
  shuffle(int times);
  int
  checkBlankString(string line);
  int
  writeNgram(ioFile* iof);
  //
  float
  computePerplexity();
  int
  readBiNgram(ioFile* iof);
  int
  readAllClassBiNgram(ioFile* iof);

};

