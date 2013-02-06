/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 BLinear Layer used only for RRLinear, RLinear, recurrent like model.
 Biases for each example in block are different
 (because it is the copy of vector of previous words)
 *******************************************************************/

class BLinear : public Module
{
public:
  floatTensor input;
  outils* otl;
  BLinear(int inputSize, int outputSize, int blockSize, outils* otl);
  ~BLinear();
  void
  reset();
  void
  changeBlockSize(int blockSize);
  floatTensor&
  forward(floatTensor& input);
  floatTensor&
  backward(floatTensor& gradOutput);
  void
  updateParameters(float learningRate);
  void
  read(ioFile *iof);
  void
  write(ioFile * iof);

};

