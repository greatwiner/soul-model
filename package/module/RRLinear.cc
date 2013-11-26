/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Recurrent Recurrent Linear  Layer, for recurrent models
 *******************************************************************/
#include "mainModule.H"
RRLinear::RRLinear(int inputSize, int blockSize, int n, string nonLinearType,
    int share, outils* otl)
{
  this->firstTime = 0;
  this->blockSize = blockSize;
  this->iContext.resize(blockSize, 1);
  this->n = n;
  BPTT = n - 1;

  this->inputSize = inputSize;
  this->nonLinearType = nonLinearType;
  this->share = share;
  this->otl = otl;
  weight.resize(inputSize, inputSize);
  weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  cstInput.resize(inputSize, blockSize);
  lastInput.resize(inputSize, blockSize);
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
  modules = new Module*[(n - 1) * step];
  modules1 = new Module*[(n - 1) * step];

  Module* module;
  BLinear* lModule;
  Module* module1;
  BLinear* lModule1;

  for (i = 0; i < n - 1; i++)
    {
      lModule = new BLinear(inputSize, inputSize, blockSize, otl);
      lModule1 = new BLinear(inputSize, inputSize, 1, otl);
      if (share)
        {
          lModule->shareWeight(weight);
          lModule1->shareWeight(weight);
        }
      add(lModule, lModule1);
      if (nonLinearType == TANH)
        {
          module = new Tanh(inputSize, blockSize); // non linear
          module1 = new Tanh(inputSize, 1);
          add(module, module1);
        }
      else if (nonLinearType == SIGM)
        {
          module = new Sigmoid(inputSize, blockSize); // non linear
          module1 = new Sigmoid(inputSize, 1);
          add(module, module1);
        }
    }
  //Three lines below create the link between input of one module with output of previous one
  firstTime = 1;
  // for test
  //cout << "RRLinear::RRLinear forward from lastInput" << endl;
  //cout << "RRLinear::RRLinear lastInput: " << endl;
  forward(lastInput);
  firstTime = 0;
}

RRLinear::~RRLinear()
{
  for (int idel = 0; idel < size; idel++)
    {
      delete modules[idel];
      delete modules1[idel];
    }
  delete[] modules;
  delete[] modules1;
}

void
RRLinear::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  this->iContext.resize(blockSize, 1);
  int inputSize = cstInput.size[0];
  floatTensor initRealTensor;
  initRealTensor.resize(cstInput.size[1], 1);
  floatTensor initSelectWeight;
  cstInput.resize(inputSize, blockSize);
  lastInput.resize(inputSize, blockSize);
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
  //Three lines below create the link between input of one module with output of previous one
  firstTime = 1;
  forward(lastInput);
  firstTime = 0;
}

void
RRLinear::add(Module* module, Module* module1)
{
  size += 1;
  modules[size - 1] = module;
  modules1[size - 1] = module1;
  output = module->output;
}

floatTensor&
RRLinear::forward(floatTensor& input)
{
	// for test
	//cout << "RRLinear::forward firstTime: " << firstTime << endl;
  if (firstTime == 2) // Discontinuous case
    {
      int i;
      int rBlockSize;
      floatTensor currentOutput1 = vectorInput;
      for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++)
        {
          if (iContext(rBlockSize) == 1)
            {
              selectOutput.select(lastInput, 1, rBlockSize);
              selectOutput.copy(vectorInput);
            }
        }
      //copy input to bias
      for (i = 0; i < n - 1; i++)
        {
          modules1[i * step]->bias.copy(input);
        }
      for (i = 0; i < size; i++)
        {
          currentOutput1 = modules1[i]->forward(currentOutput1);
        }

      for (i = 0; i < size; i++)
        {
          for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++)
            {
              if (iContext(rBlockSize) == 1)
                {
                  selectOutput.select(modules[i]->output, 1, rBlockSize);
                  selectOutput.copy(modules1[i]->output);
                }
            }
        }
      firstTime = 0;
      return currentOutput;
    }
  else if (firstTime == 1)
    {
      int i;
      //copy input to bias
      for (i = 0; i < n - 1; i++)
        {

          subInput.sub(input, inputSize * i, inputSize * (i + 1) - 1, 0,
              blockSize - 1);
          modules[i * step]->bias.copy(subInput);
        }
      currentOutput = cstInput;
      lastInput.copy(cstInput);
      for (i = 0; i < size; i++)
        {
          currentOutput = modules[i]->forward(currentOutput);
        }
      firstTime = 0;
      return currentOutput;
    }
  else if (firstTime == 0)
    {
      int i;
      //copy input to bias
      for (i = 0; i < n - 1; i++)
        {
          subInput.sub(input, inputSize * i, inputSize * (i + 1) - 1, 0,
              blockSize - 1);
          modules[i * step]->bias.copy(subInput);
        }
      lastInput.copy(modules[step - 1]->output);
      for (i = 0; i < size - step; i++)
        {
          modules[i]->output.copy(modules[i + step]->output);
        }
      currentOutput = modules[size - step - 1]->output;
      for (i = size - step; i < size; i++)
        {
          currentOutput = modules[i]->forward(currentOutput);
        }
      return currentOutput;
    }
  return currentOutput;
}
floatTensor&
RRLinear::backward(floatTensor& gradOutput)
{
  int updateBPTT = BPTT;
  //updateBPTT++;
  currentGradOutput = gradOutput;
  Module* currentModule = modules[size - 1];
  Module* previousModule;
  int i;
  if (updateBPTT % BPTT == 0)
    {
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
          subGradInput.sub(gradInput, inputSize * i, inputSize * (i + 1) - 1,
              0, blockSize - 1);
          subGradInput.copy(modules[i * step]->gradOutput);
        }
    }
  else
    {
      for (i = size - 2; i > size - 2 - step; i--)
        {
          previousModule = modules[i];
          currentGradOutput = currentModule->backward(currentGradOutput);
          currentModule = previousModule;
        }
      //copy gradBias and return it as input bias
      gradInput = 0;
      for (i = n - 2; i < n - 1; i++) // BPTT = 1
        {
          subGradInput.sub(gradInput, inputSize * i, inputSize * (i + 1) - 1,
              0, blockSize - 1);
          subGradInput.copy(modules[i * step]->gradOutput);
        }
    }
  return gradInput;
}

void
RRLinear::updateParameters(float learningRate)
{
  int i;
  int updateBPTT = BPTT;
  if (share)
    {
      modules[(size - step) - step * (BPTT - 1)]->weightDecay = weightDecay;
    }
  else
    {
      for (i = (size - step) - step * (BPTT - 1); i < size; i += step) // BPTT with maxable T n - 1
        {
          modules[i]->weightDecay = weightDecay;
        }
    }
  if (updateBPTT % BPTT == 0)
    {

      for (i = (size - step) - step * (BPTT - 1); i < size; i += step) // BPTT with maxable T n - 1
        {
          modules[i]->updateParameters(learningRate);
        }
    }
  else
    {
      for (i = (size - step); i < size; i += step) // BPTT = 1
        {
          modules[i]->updateParameters(learningRate);
        }
    }
}
float
RRLinear::distance2(Module& anotherRRLinear) {
	floatTensor distMatrix;
	distMatrix.copy(this->vectorInput);
	distMatrix.axpy(anotherRRLinear.vectorInput, -1);
	float res1 = distMatrix.sumSquared();
	if (res1 > 0) {
		cout << "RRLinear::distance2 res1 > 0" << endl;
	}
	distMatrix.resize(this->weight);
	distMatrix.copy(this->weight);
	distMatrix.axpy(anotherRRLinear.weight, -1);
	return res1+distMatrix.sumSquared();
}
void
RRLinear::read(ioFile* iof)
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
  weight.read(iof);
}
void
RRLinear::write(ioFile* iof)
{
  iof->writeString((char*) "RRLinear");
  vectorInput.write(iof);
  weight.write(iof);
}
