/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Recurrent Recurrent Linear  Layer, for recurrent models
 *******************************************************************/
class RRLinear : public Module
{
public:
  floatTensor gradOutput; // pointer used for  backward and update
  //floatTensor gradOutputT; //
  floatTensor input; // pointer used for  backward and update
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
  floatTensor lastInput;
  int inputSize;
  Module** modules; // the arrays of modules Linears + Sigmoids (Tanhs)
  Module** modules1; // the arrays of modules Linears + Sigmoids (Tanhs)
  RRLinear(int inputSize, int blockSize, int n, string nonLinearType,
      int share, outils* otl);
  ~RRLinear();
  int share;
  /* init. weights and bias */
  void
  reset();
  void
  changeBlockSize(int blockSize);

  floatTensor currentOutput;
  floatTensor currentGradOutput;
  floatTensor selectOutput;
  void
  add(Module* module, Module* module1);

  floatTensor&
  forward(floatTensor& input);
  floatTensor&
  backward(floatTensor& gradOutput);
  void
  updateParameters(float learningRate);
  float
  distance2(RRLinear& anotherRRLinear);
  void
  read(ioFile *iof);
  void
  write(ioFile * iof);

};
