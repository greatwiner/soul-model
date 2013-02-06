/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Specific module used to build a sequential network.  The methods
 forward, backward, and updateParameters only call sequentially the
 corresponding methods of each module. The add method aims to add a
 new module, and the last added module is the output (the first
 added module is the input of the network).
 *******************************************************************/
#include "mainModule.H"
Sequential::Sequential(int maxSize)
{
  size = 0;
  modules = new Module*[maxSize];
}
Sequential::~Sequential()
{
  delete lkt;
  for (int idel = 0; idel < size; idel++)
    {
      delete modules[idel];
    }
  delete[] modules;
}
void
Sequential::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  int i;
  for (i = 0; i < size; i++)
    {
      modules[i]->changeBlockSize(blockSize);
    }
  lkt->changeBlockSize(blockSize);
  // Update output
  output = modules[size - 1]->output;
}

void
Sequential::add(Module* module)
{
  size++;
  modules[size - 1] = module;
  output = module->output;
}

floatTensor&
Sequential::forward(intTensor& input)
{
  int i;
  currentOutput = lkt->forward(input);
  for (i = 0; i < size; i++)
    {
      currentOutput = modules[i]->forward(currentOutput);
    }
  return currentOutput;
}

floatTensor&
Sequential::backward(floatTensor& gradOutput)
{
  currentGradOutput = gradOutput;

  Module* currentModule = modules[size - 1];
  Module* previousModule;
  int i;
  for (i = size - 2; i > -1; i--)
    {
      previousModule = modules[i];
      currentGradOutput = currentModule->backward(currentGradOutput);
      currentModule = previousModule;
    }
  currentGradOutput = currentModule->backward(currentGradOutput);
  // Also backward with Lookup Table
  currentGradOutput = lkt->backward(currentGradOutput);
  return currentGradOutput;
}

void
Sequential::updateParameters(float learningRate)
{
  int i;
  for (i = 0; i < size; i++)
    {
      modules[i]->updateParameters(learningRate);
    }
  // Also update with Lookup Table
  lkt->updateParameters(learningRate);
}

void
Sequential::read(ioFile* iof)
{
  int i;
  lkt->read(iof);
  for (i = 0; i < size; i++)
    {
      modules[i]->read(iof);
    }
}
void
Sequential::write(ioFile* iof)
{
  int i;
  lkt->write(iof);
  for (i = 0; i < size; i++)
    {
      modules[i]->write(iof);
    }
}

