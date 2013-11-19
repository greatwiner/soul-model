/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Specific class for language model. This layer aims to project context
 words to a space, then concatenate them to create a new layer
 Input is the indices of context words (n - 1 integers)
 Output is the vector after concatenation (n - 1) x dimensionSize floats
 *******************************************************************/

class LookupTable : public Embeddings
{
public:

  LookupTable();
  LookupTable(int nIndex, int nDimension, int inputSize, int blockSize,
      int oneClass, outils* otl);

  ~LookupTable();
  void
  reset();
  void
  updateParameters(float learningRate);
  void
  read(ioFile *iof);
  void
  write(ioFile * iof);
};

