/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Linear Layer.
 *******************************************************************/
#include "mainModule.H"

Linear::Linear(int inputSize, int outputSize, int blockSize, outils* otl)
{
  // Initialize parameters
  this->blockSize = blockSize;
  weightDecay = 0;
  weight.resize(inputSize, outputSize);
  bias.resize(outputSize, 1);
  V1col.resize(blockSize, 1);
  V1col = 1;
  gradInput.resize(inputSize, blockSize);
  output.resize(outputSize, blockSize);

  this->otl = otl;
  reset();
}

Linear::~Linear()
{
}

void
Linear::changeBlockSize(int blockSize)
{
  // Need to change memory size for some parameters
  this->blockSize = blockSize;
  int inputSize = gradInput.size[0];
  int outputSize = output.size[0];
  V1col.resize(blockSize, 1);
  V1col = 1;
  gradInput.resize(inputSize, blockSize);
  output.resize(outputSize, blockSize);

}

void
Linear::reset()
{
  weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  bias.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
}

floatTensor&
Linear::forward(floatTensor& input)
{
  this->input = input;
  output = 0;
  // output = block_size bias columns
  output.ger(bias, V1col, 1);

  // output = weight^T x input
  output.gemm(weight, 'T', input, 'N', 1, 1);
  return output;
}

floatTensor&
Linear::backward(floatTensor& gradOutput)
{
	// for test
	//cout << "Linear::backward" << endl;
  // Keep gradOutput for later update
  this->gradOutput = gradOutput;

  // gradInput = weight x gradOutput
  gradInput.gemm(weight, 'N', gradOutput, 'N', 1, 0);
  return gradInput;
}

void
Linear::updateParameters(float learningRate)
{
	// for test
	//cout << "Linear::updateParameters" << endl;
  // weight = - learningRate x input x gradOutput^T
  //        + weight - learningRate * weightDecay * weight
  weight.gemm(input, 'N', gradOutput, 'T', -learningRate,
      1 - learningRate * weightDecay);
  // bias = -learningRate x gradOutput x V1col + bias
  bias.gemv(gradOutput, 'N', V1col, -learningRate, 1);
}

void
Linear::read(ioFile* iof)
{
  iof->readString(name);
  weight.read(iof);
  bias.read(iof);
}
void
Linear::write(ioFile* iof)
{
  iof->writeString((char*) "Linear");
  weight.write(iof);
  bias.write(iof);
}
