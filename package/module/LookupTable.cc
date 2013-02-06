/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Specific class for language model. This layer aims to project context
 words to a space, then concatenate them to create a new layer
 Input is the indices of context words (n - 1 integers)
 Output is the vector after concatenation (n - 1) x dimensionSize floats
 *******************************************************************/

#include "mainModule.H"

LookupTable::LookupTable(int indexNumber, int dimensionSize, int inputSize,
    int blockSize, int oneClass, outils* otl)
{
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
}

void
LookupTable::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  int size0 = output.size[0];
  output.resize(size0, blockSize);
}
void
LookupTable::reset()
{
  weight.uniform(LKT_INIT0, LKT_INIT1, otl);
}

void
LookupTable::init1class()
{
  floatTensor initRealTensor;
  initRealTensor.resize(weight.size[0], 1);
  initRealTensor.uniform(LKT_INIT0, LKT_INIT1, otl);
  floatTensor initSelectWeight;
  int i;
  for (i = 0; i < weight.size[1]; i++)
    {
      initSelectWeight.select(weight, 1, i);
      initSelectWeight.copy(initRealTensor);
    }
}
floatTensor&
LookupTable::forward(floatTensor& input)
{
  cout << "Wrong call, must call with input is intTensor" << endl;
  return input;
}

floatTensor&
LookupTable::forward(intTensor& input)
{
  this->input = input;
  int x0, x1;
  for (int i = 0; i < input.size[1]; i++) // with blockSize
    {
      x0 = 0;
      x1 = dimensionSize - 1;
      for (int j = 0; j < input.size[0]; j++) // with context word number
        {
          selectOutput.sub(output, x0, x1, i, i);
          selectWeight.select(weight, 1, input(j, i));
          selectOutput.copy(selectWeight);
          x0 += dimensionSize;
          x1 += dimensionSize;
        }
    }
  return output;
}

floatTensor&
LookupTable::backward(floatTensor& gradOutput)
{
  gradWeight = gradOutput;
  // Don't use return variable, so whatever you want :S
  return gradWeight;
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
  iof->readString(name);
  weight.read(iof);
}
void
LookupTable::write(ioFile* iof)
{
  iof->writeString((char*) "LookupTable");
  weight.write(iof);
}
