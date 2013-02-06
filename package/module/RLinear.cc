/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Recurrent Linear  Layer, for pseudo-recurrent n-gram models.
 *******************************************************************/
#include "mainModule.H"
RLinear::RLinear(int inputSize, int blockSize, int n, string nonLinearType,
    int share, outils* otl)
{
  this->blockSize = blockSize;
  this->n = n;
  BPTT = n - 1;
  this->inputSize = inputSize;
  this->nonLinearType = nonLinearType;
  this->share = share;
  this->otl = otl;
  weight.resize(inputSize, inputSize);
  weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  cstInput.resize(inputSize, blockSize);
  vectorInput.resize(inputSize, 1);
  vectorInput.uniform(LKT_INIT0, LKT_INIT1, otl);
  floatTensor initSelectInput;
  int i;
  for (i = 0; i < blockSize; i++)
    {
      initSelectInput.select(cstInput, 1, i);
      initSelectInput.copy(vectorInput);
    }
  gradInput.resize(inputSize * (n - 1), blockSize);
  size = 0;

  step = 1;
  if (nonLinearType == TANH)
    {
      step = 2;
    }
  else if (nonLinearType == SIGM)
    {
      step = 2;
    }
  modules = new Module*[(n - 2) * step + 1];

  Module* module;
  BLinear* lModule;
  for (i = 0; i < n - 2; i++)
    {
      lModule = new BLinear(inputSize, inputSize, blockSize, otl);
      if (share)
        {
          lModule->shareWeight(weight);
        }
      add(lModule);
      if (nonLinearType == TANH)
        {
          module = new Tanh(inputSize, blockSize); // non linear
          add(module);
        }
      else if (nonLinearType == SIGM)
        {
          module = new Sigmoid(inputSize, blockSize); // non linear
          add(module);
        }
    }
  //Last don't add Sigmoid, Tanh, add in NgramModel
  lModule = new BLinear(inputSize, inputSize, blockSize, otl);
  if (share)
    {
      lModule->shareWeight(weight);
    }
  add(lModule);

}

RLinear::~RLinear()
{
  for (int idel = 0; idel < size; idel++)
    {
      delete modules[idel];
    }
  delete[] modules;
}

void
RLinear::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  int inputSize = cstInput.size[0];
  floatTensor initSelectWeight;
  cstInput.resize(inputSize, blockSize);
  int i;
  for (i = 0; i < cstInput.size[1]; i++)
    {
      initSelectWeight.select(cstInput, 1, i);
      initSelectWeight.copy(vectorInput);
    }

  gradInput.resize(inputSize * (n - 1), blockSize);
  for (i = 0; i < size; i++)
    {
      modules[i]->changeBlockSize(blockSize);
    }
  output = modules[size - 1]->output;
}

void
RLinear::add(Module* module)
{
  size += 1;
  modules[size - 1] = module;
  output = module->output;
}

floatTensor&
RLinear::forward(floatTensor& input)
{
  int i;
  for (i = 0; i < n - 1; i++)
    {
      subInput.sub(input, inputSize * i, inputSize * (i + 1) - 1, 0,
          blockSize - 1);
      modules[i * step]->bias.copy(subInput);
    }
  currentOutput = cstInput;
  for (i = 0; i < size; i++)
    {
      currentOutput = modules[i]->forward(currentOutput);
    }
  return currentOutput;
}

floatTensor&
RLinear::backward(floatTensor& gradOutput)
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
  //copy gradBias and return it as input bias
  gradInput = 0;
  for (i = n - 1 - BPTT; i < n - 1; i++) // BPTT with maxable T n - 1
    {
      subGradInput.sub(gradInput, inputSize * i, inputSize * (i + 1) - 1, 0,
          blockSize - 1);
      subGradInput.copy(modules[i * step]->gradOutput);
    }
  return gradInput;
}

void
RLinear::updateParameters(float learningRate)
{
  int i;
  if (share)
    {
      modules[(size - 1) - step * (BPTT - 1)]->weightDecay = weightDecay;
    }
  else
    {
      for (i = (size - 1) - step * (BPTT - 1); i < size; i += step) // BPTT with maxable T n - 1
        {
          modules[i]->weightDecay = weightDecay;
        }
    }
  for (i = (size - 1) - step * (BPTT - 1); i < size; i += step) // BPTT with maxable T n - 1
    {
      modules[i]->updateParameters(learningRate);
    }
}
void
RLinear::read(ioFile* iof)
{
  iof->readString(name);
  vectorInput.read(iof);
  floatTensor initSelectInput;
  int i;
  for (i = 0; i < blockSize; i++)
    {
      initSelectInput.select(cstInput, 1, i);
      initSelectInput.copy(vectorInput);
    }
  if (!share)
    {
      for (int i = 0; i < size; i += step)
        {
          modules[i]->weight.read(iof);
        }
    }
  else
    {
      weight.read(iof);
    }
}
void
RLinear::write(ioFile* iof)
{
  iof->writeString((char*) "RLinear");
  vectorInput.write(iof);
  if (!share)
    {
      for (int i = 0; i < size; i += step)
        {
          modules[i]->weight.write(iof);
        }
    }
  else
    {
      weight.write(iof);
    }
}
