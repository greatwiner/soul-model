/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Specific class for language model. This layer aims to be the softmax
 layer which takes as input the last hidden layer to predict
 word (class) probabilities.
 Basically, this code represent a Linear Layer  join with a softmax
 Ask Linear.H for more
 *******************************************************************/
class LinearSoftmax : public ProbOutput
{
public:
  LinearSoftmax();
  LinearSoftmax(int inputSize, int outputSize, int blockSize, outils* otl);
  ~LinearSoftmax();
  void
  reset();
  floatTensor&
  backward(intTensor& word, floatTensor& coefTensor);
  void
  updateParameters(float learningRate);
  void
  read(ioFile *iof);
  void
  write(ioFile * iof);
};

