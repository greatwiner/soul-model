/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Specific class for language model. This layer aims to project context
 words to a space, then concatenate them to create a new layer
 Input is the indices of context words (n - 1 integers)
 Output is the vector after concatenation (n - 1) x dimensionSize floats
 *******************************************************************/

#include "mainModule.H"

LookupTable::LookupTable() {

}

LookupTable::LookupTable(int indexNumber, int dimensionSize, int inputSize,
    int blockSize, int oneClass, outils* otl)
{
	name = "LookupTable";
  weight.resize(dimensionSize, indexNumber);
  output.resize(dimensionSize * inputSize, blockSize);
  this->otl = otl;
  this->blockSize = blockSize;
  if (!oneClass)
    {
      reset();
    }
  else
    {
      init1class();
    }
  this->dimensionSize = dimensionSize;
  this->indexNumber = indexNumber;
}

LookupTable::~LookupTable()
{
	// for test
	//cout << "LookupTable::~LookupTable here" << endl;
}

void
LookupTable::reset()
{
  weight.uniform(LKT_INIT0, LKT_INIT1, otl);
}

void
LookupTable::updateParameters(float learningRate)
{
  int x0, x1;
  for (int i = 0; i < input.size[1]; i++)
    {
      x0 = 0;
      x1 = dimensionSize - 1;
      for (int j = 0; j < input.size[0]; j++)
        {
          selectWeight.select(weight, 1, input(j, i));
          selectGradWeight.sub(gradWeight, x0, x1, i, i);
          if (weightDecay != 0)
            {
              // y = y - lr * wd * y
              selectWeight.scal(1 - learningRate * weightDecay);
            }
          selectWeight.axpy(selectGradWeight, -learningRate);
          x0 += dimensionSize;
          x1 += dimensionSize;
        }
    }
}

void
LookupTable::read(ioFile* iof)
{
	// for test
	//cout << "LookupTable::read here" << endl;
  iof->readString(name);
  // for test
  //cout << "LookupTable::read name: " << name << endl;
  // for test
  //cout << "LookupTable::read shareW: " << shareW << endl;
  weight.read(iof);
  // for test
  //cout << "LookupTable::read here 1" << endl;
}
void
LookupTable::write(ioFile* iof)
{
  iof->writeString(name);
  weight.write(iof);
  if (name == "LookupTable_AG") {
	  // for test
	  cout << "LookupTable::write here" << endl;
	  floatTensor cumulGradWeight(1, indexNumber);
	  cumulGradWeight = INIT_VALUE_ADAG;
	  cumulGradWeight.write(iof);
  }
}
