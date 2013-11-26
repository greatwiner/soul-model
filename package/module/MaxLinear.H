/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Max Linear Layer, used after LookupTable to have a max function
 at the hidden layer.

 output = element wise max over position (weight_i x R^Tv_i) + bias,
 v_i is the index of the ith context word,
 weight_i is a submatrix of weight for position i


 See Module.H for more detail, for example:
 weight and bias are declared in Module.H
 *******************************************************************/

class MaxLinear : public Module
{
public:

  floatTensor input;
  floatTensor internalOutput;
  floatTensor internalGradOutput;
  intTensor active;
  floatTensor V1col; // For the bunch mode.
  floatTensor subWeight;
  floatTensor subInput;
  floatTensor subInternalOutput;
  floatTensor copySubInternalOutput;
  floatTensor copySubInput;
  outils* otl;
  MaxLinear(int elementNumber, int inputSize, int outputSize, int blockSize,
      outils* otl);
  ~MaxLinear();

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
  float
  distance2(Module& anotherModule);

  int elementNumber;
  int inputSize;
  int outputSize;
};

