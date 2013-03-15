/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Linear Layer. May be plugged with a non-linear activity function,
 such as Tanh or Sighmoid.
 There is LinearSoftmax: Concatenation of Linear and Softmax.

 weight = output_size x input_size
 output = weight^T x input

 See Module.H for more detail, for example:
 weight and bias are declared in Module.H
 *******************************************************************/

class Linear : public Module
{
public:
  // Pointer for the output of the previous layer, used for updateParameters
  floatTensor input;

  // For the bunch mode, copy bias vector for all columns of
  // (floatTensor) output using matrix multiplication
  floatTensor V1col;

  // Pointer to a random generator
  outils* otl;

  Linear(int inputSize, int outputSize, int blockSize, outils* otl);

  ~Linear();
  // Initialize parameters:  weights and bias
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

