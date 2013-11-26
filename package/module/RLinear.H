/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Recurrent Linear  Layer, for pseudo-recurrent n-gram models.
 *******************************************************************/

class RLinear : public Module
{
public:
  floatTensor gradOutput; // pointer used for  backward and update
  floatTensor V1col; // For the bunch mode.
  outils* otl;
  int n;
  int size;
  int step;
  int BPTT;
  string nonLinearType;
  floatTensor cstInput;
  floatTensor subInput;
  floatTensor subGradInput;
  int inputSize;
  RLinear(int inputSize, int blockSize, int n, string nonLinearType, int share,
      outils* otl);
  ~RLinear();
  int share;
  /* init. weights and bias */
  void
  reset();
  void
  changeBlockSize(int blockSize);
  Module** modules; // the arrays of modules Linears + Sigmoids (Tanhs)
  floatTensor currentOutput;
  floatTensor currentGradOutput;
  void
  add(Module* module);

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

};

