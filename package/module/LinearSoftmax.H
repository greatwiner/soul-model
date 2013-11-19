/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Specific class for language model. This layer aims to be the softmax
 layer which takes as input the last hidden layer to predict
 word (class) probabilities.
 Basically, this code represent a Linear Layer  join with a softmax
 Ask Linear.H for more
 *******************************************************************/
class LinearSoftmax : public Module
{
public:
  /*same as Linear */
  floatTensor gradOutput;
  floatTensor input;
  floatTensor softmaxVCol;
  floatTensor V1col;
  floatTensor softmaxV1row;

  /* Specific, only used for bunch mode computation*/
  floatTensor selectGradOutput;
  floatTensor preOutput;
  floatTensor selectOutput;
  floatTensor selectPreOutput;
  outils* otl;
  LinearSoftmax();
  LinearSoftmax(int inputSize, int outputSize, int blockSize, outils* otl);
  ~LinearSoftmax();
  void
  reset();
  void
  changeBlockSize(int blockSize);
  floatTensor&
  forward(floatTensor& input);
  floatTensor&
  backward(floatTensor& word);
  floatTensor&
  backward(intTensor& word);
  void
  updateParameters(float learningRate);
  float
  distance2(LinearSoftmax& anotherOutput);
  void
  read(ioFile *iof);
  void
  write(ioFile * iof);
};

