/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 BLinear Layer used only for RRLinear, RLinear, recurrent like model.
 Biases for each example in block are different
 (because it is the copy of vector of previous words)
 *******************************************************************/
#include "mainModule.H"

BLinear::BLinear(int inputSize, int outputSize, int blockSize, outils* otl)
{
  this->blockSize = blockSize;
  weightDecay = 0;
  weight.resize(inputSize, outputSize);
  weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  bias.resize(outputSize, blockSize);
  gradInput.resize(inputSize, blockSize);
  output.resize(outputSize, blockSize);
  this->otl = otl;
  reset();
}

BLinear::~BLinear()
{
}

void
BLinear::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  int inputSize = gradInput.size[0];
  int outputSize = output.size[0];
  gradInput.resize(inputSize, blockSize);
  output.resize(outputSize, blockSize);
  bias.resize(outputSize, blockSize);
}

void
BLinear::reset()
{
  weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  bias.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
}

floatTensor&
BLinear::forward(floatTensor& input)
{
  this->input = input;
  output.copy(bias);
  output.gemm(weight, 'T', input, 'N', 1, 1);
  return output;
}

floatTensor&
BLinear::backward(floatTensor& gradOutput)
{
  this->gradOutput = gradOutput;
  gradInput.gemm(weight, 'N', gradOutput, 'N', 1, 0);
  return gradInput;
}

void
BLinear::updateParameters(float learningRate)
{
  weight.gemm(input, 'N', gradOutput, 'T', -learningRate,
      1 - learningRate * weightDecay);
}

void
BLinear::read(ioFile* iof)
{
  iof->readString(name);
  weight.read(iof);
  bias.read(iof);
}
void
BLinear::write(ioFile* iof)
{
  iof->writeString((char*) "BLinear");
  weight.write(iof);
  bias.write(iof);
}

float
BLinear::distance2(Module& anotherModule) {
	// TODO
	return 0;
}
