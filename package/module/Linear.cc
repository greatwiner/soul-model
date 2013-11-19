/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Linear Layer.
 *******************************************************************/
#include "mainModule.H"

Linear::Linear() {

}

Linear::Linear(int inputSize, int outputSize, int blockSize, outils* otl)
{
	name = "Linear";
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
  int inputSize = gradInput.getSize(0);
  //int inputSize = gradInput.size[0];
  int outputSize = output.getSize(0);
  //int outputSize = output.size[0];
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

  output.gemm(weight, 'T', input, 'N', 1, 1);
  /*if (output.testNan() != 0) {
	  cout << "Linear::forward output is nan" << endl;
	  exit(0);
  }*/
  return output;
}

floatTensor&
Linear::backward(floatTensor& gradOutput)
{
  // Keep gradOutput for later update
  this->gradOutput = gradOutput;

  // gradInput = weight x gradOutput
  gradInput.gemm(weight, 'N', gradOutput, 'N', 1, 0);
  /*if (gradInput.testNan() != 0) {
	  cout << "Linear::backward gradInput is nan" << endl;
	  cout << "Linear::backward gradInput: " << endl;
	  gradInput.write();
	  if (weight.testNan() != 0) {
		  cout << "Linear::backward because weight is nan" << endl;
	  }
	  else {
		  cout << "Linear::backward weight is normal: " << endl;
		  weight.write();
	  }
	  if (gradOutput.testNan() != 0) {
		  cout << "Linear::backward because gradOutput is nan" << endl;
	  }
	  else {
		  cout << "Linear::backward gradOutput is normal" << endl;
		  gradOutput.write();
	  }
	  cout << "Linear::backward program will exit" << endl;
	  exit(0);
  }*/
  return gradInput;
}

void
Linear::updateParameters(float learningRate)
{
  // weight = - learningRate x input x gradOutput^T
  //        + weight - learningRate * weightDecay * weight
  weight.gemm(input, 'N', gradOutput, 'T', -learningRate,
      1 - learningRate * weightDecay);
  // bias = -learningRate x gradOutput x V1col + bias
  bias.gemv(gradOutput, 'N', V1col, -learningRate, 1);
}

float
Linear::distance2(Linear& anotherLinear) {
	floatTensor distMatrix;
	distMatrix.copy(this->weight);
	distMatrix.axpy(anotherLinear.weight, -1);
	float res1 = distMatrix.sumSquared();
	distMatrix.resize(this->bias);
	distMatrix.copy(this->bias);
	distMatrix.axpy(anotherLinear.bias, -1);
	return res1+distMatrix.sumSquared();
}

void
Linear::read(ioFile* iof)
{
	// for test
	//cout << "Linear::read here" << endl;
  iof->readString(name);
  // for test
  //cout << "Linear::read name: " << name << endl;
  weight.read(iof);
  // for test
  //cout << "Linear::read here 1" << endl;
  bias.read(iof);
  // for test
  //cout << "Linear::read here 2" << endl;
}
void
Linear::write(ioFile* iof)
{
  iof->writeString(name);
  // for test
  cout << "Linear::write name: " << name << endl;
  weight.write(iof);
  bias.write(iof);
  if (name == "Linear_AG") {
	  // for test
	  cout << "Linear::write here" << endl;
	  iof->writeFloat(INIT_VALUE_ADAG);
	  iof->writeFloat(INIT_VALUE_ADAG);
  }
}
