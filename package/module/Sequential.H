/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Specific module used to build a sequential network.  The methods
 forward, backward, and updateParameters only call sequentially the
 corresponding methods of each module. The add method aims to add a
 new module, and the last added module is the output (the first
 added module is the input of the network).
 *******************************************************************/

class Sequential
{
public:
  // output: points to the output of the last module
  floatTensor output;

  // for the bunch mode
  int blockSize;

  // the number of modules without LookupTable
  int size;

  // the arrays of modules except LookupTable
  Module** modules;

  // Here is LookupTable
  Embeddings* lkt;

  // Tensor for forward and backward pass
  floatTensor currentOutput;
  floatTensor currentGradOutput;

  Sequential(int maxSize);
  ~Sequential();

  // Change block size for all modules and LookupTable
  void
  changeBlockSize(int blockSize);

  // Add one module
  void
  add(Module* module);

  floatTensor&
  forward(intTensor& input);
  floatTensor&
  backward(floatTensor& gradOutput);
  void
  updateParameters(float learningRate);
  void
  read(ioFile *iof);
  void
  write(ioFile * iof);
  void
  write(ioFile* iof, string name);
};

