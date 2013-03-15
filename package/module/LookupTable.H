/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Specific class for language model. This layer aims to project context
 words to a space, then concatenate them to create a new layer
 Input is the indices of context words (n - 1 integers)
 Output is the vector after concatenation (n - 1) x dimensionSize floats
 *******************************************************************/

class LookupTable : public Module
{
public:
  floatTensor gradWeight;
  intTensor input;
  // tensor used to select some part of other tensors, act as a pointer
  floatTensor selectOutput;
  floatTensor selectWeight;
  floatTensor selectGradWeight;

  // projection dimension
  int dimensionSize;

  // number of indices ~ number of words in the vocabulary
  int indexNumber;

  outils* otl;
  LookupTable(int nIndex, int nDimension, int inputSize, int blockSize,
      int oneClass, outils* otl);

  ~LookupTable();
  void
  reset();
  void
  changeBlockSize(int blockSize);
  floatTensor&
  forward(floatTensor& input);
  floatTensor&
  backward(floatTensor& gradOutput);
  floatTensor&
  forward(intTensor& input);
  void
  updateParameters(float learningRate);
  void
  read(ioFile *iof);
  void
  write(ioFile * iof);
  // Initialize with only one vector for all words
  void
  init1class();
};

