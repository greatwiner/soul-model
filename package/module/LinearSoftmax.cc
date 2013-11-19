/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Specific class for language model. This layer aims to be the softmax
 layer which takes as input the last hidden layer to predict
 word (class) probabilities.
 *******************************************************************/
#include "mainModule.H"

LinearSoftmax::LinearSoftmax() {

}

LinearSoftmax::LinearSoftmax(int inputSize, int outputSize, int blockSize,
    outils* otl)
{
	name = "LinearSoftmax";
  this->blockSize = blockSize;
  weightDecay = 0;
  weight.resize(inputSize, outputSize);
  bias.resize(outputSize, 1);
  V1col.resize(blockSize, 1);
  V1col = 1;
  softmaxV1row.resize(outputSize, 1);
  softmaxV1row = 1;
  softmaxVCol.resize(blockSize, 1);
  gradInput.resize(inputSize, blockSize);
  output.resize(outputSize, blockSize);
  gradOutput.resize(output);
  preOutput.resize(outputSize, blockSize);

  this->otl = otl;
  reset();
}

LinearSoftmax::~LinearSoftmax()
{
}

void
LinearSoftmax::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  V1col.resize(blockSize, 1);
  V1col = 1;
  int inputSize = gradInput.size[0];
  int outputSize = output.size[0];
  softmaxVCol.resize(blockSize, 1);
  gradInput.resize(inputSize, blockSize);
  output.resize(outputSize, blockSize);
  gradOutput.resize(output);
  preOutput.resize(outputSize, blockSize);

}

void
LinearSoftmax::reset()
{
  weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  bias.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
}
floatTensor&
LinearSoftmax::forward(floatTensor& input)
{
  this->input = input;
  // preOutput is the same as output in Linear.cc
  preOutput = 0;
  // preOutput = block_size bias columns
  preOutput.ger(bias, V1col, 1);

  // Then, preOutput = preOutput + weight^T x input
  preOutput.gemm(weight, 'T', input, 'N', 1, 1);

  // for test
  //cout << "LinearSoftmax::forward here" << endl;

  // For each column, minus minimum value
  for (int i = 0; i < output.size[1]; i++)
    {
      float max = -10000000;
      float min = 10000000;
      for (int j = 0; j < output.size[0]; j++) {
		  if (preOutput(j, i) > max) {
              max = preOutput(j, i);
          }
		  if (preOutput(j ,i) < min) {
			  min = preOutput(j, i);
		  }
      }
      for (int j = 0; j < output.size[0]; j++) {
          preOutput(j, i) -= (min+max)/2;
      }
    }
  // for test
  //cout << "LinearSoftmax::forward here1" << endl;
  // output = e^(preOutput)
  /*if (output.testInf() != 0) {
	  cout << "LinearSoftmax::forward output is inf before" << endl;
  	  exit(0);
  }*/
  output.mexp(preOutput);

  // softmaxVCol contains the sum for each column
  softmaxVCol.gemv(output, 'T', softmaxV1row, 1, 0);
  /*if (output.testInf() != 0) {
	  cout << "LinearSoftmax::forward output is inf" << endl;
	  if (preOutput.testInf() != 0) {
		  cout << "LinearSoftmax::forward because preOutput is inf" << endl;
	  }
	  else {
		  cout << "LinearSoftmax::forward preOutput: " << endl;
		  preOutput.write();
		  cout << "LinearSoftmax::forward preOutput colonne 39" << endl;
		  floatTensor preOutputSelect;
		  preOutputSelect.select(preOutput, 1, 39);
		  preOutputSelect.write();
		  cout << "LinearSoftmax::forward output:" << endl;
		  output.write();
	  }
	  exit(0);
  }*/

  // For each column, divide by the sum to have
  // for each element e^x_i / \sum_j e^x_j
  for (int i = 0; i < output.size[1]; i++) {
	  selectOutput.select(output, 1, i);
      // for test
      //cout << "LinearSoftmax::forward here2" << endl;
      /*if (selectOutput.testNan() != 0) {
		  cout << "LinearSoftmax::forward selectOutput is nan before" << endl;
		  exit(0);
      }*/
      selectOutput.scal(1.0 / softmaxVCol(i));
      // for test
      //cout << "LinearSoftmax::forward here3" << endl;
      /*if (selectOutput.testNan() != 0) {
    	  cout << "LinearSoftmax::forward selectOutput is nan" << endl;
    	  cout << softmaxVCol(i) << endl;
    	  exit(0);
      }*/
  }
  // for test
  //cout << "LinearSoftmax::forward here4" << endl;

  return output;
}

floatTensor&
LinearSoftmax::backward(floatTensor& word)
{
  cerr << "ERROR: backward of LinearSoftmax for floatTensor" << endl;
  exit(1);
}

floatTensor&
LinearSoftmax::backward(intTensor& word)
{
  // gradOutput for weight and bias of the Linear part,
  // computed from the output after softmax (not the output of
  // the Linear part
	// for test
	//cout << "LinearSoftmax::backward here" << endl;
  gradOutput.copy(output);
  // for test
  //cout << "LinearSoftmax::backward here1" << endl;
  int i;
  this->input = input;
  // for test
  //cout << "LinearSoftmax::backward here2" << endl;
  for (i = 0; i < blockSize; i++)
    {
      // If taking account this n-gram
      // In some cases, for some examples in the block, we don't
      // want to update with them, e.g., blockSize = 128 but in the
      // last block, we have only 78, predicted word for 50 *
      // last examples should be set to SIGN_NOT_WORD
      // If using, subtract its value in gradOutput 1
      if (word(i) != SIGN_NOT_WORD)
        {

          gradOutput(word(i), i) -= 1;
        }
      // Not use, all values = 0
      else
        {
    	  // for test
    	  //cout << "LinearSoftmax::backward here3" << endl;
          selectGradOutput.select(gradOutput, 1, i);
          selectGradOutput = 0;
        }
    }
  // for test
  //cout << "LinearSoftmax::backward here4" << endl;
  /*if (gradInput.testNan() != 0) {
	  cout << "LinearSoftmax::backward gradInput is nan before" << endl;
	  exit(0);
  }*/
  gradInput.gemm(weight, 'N', gradOutput, 'N', 1, 0);
  // for test
  //cout << "LinearSoftmax::backward here5" << endl;
  /*if (gradInput.testNan() != 0) {
	  cout << "LinearSoftmax::backward gradInput is nan" << endl;
	  cout << "LinearSoftmax::backward gradInput: " << endl;
	  gradInput.write();
	  if (weight.testNan() != 0) {
		  cout << "LinearSoftmax::backward because weight is nan" << endl;
	  }
	  else {
		  cout << "LinearSoftmax::backward weight normal" << endl;
		  weight.write();
	  }
	  if (gradOutput.testNan() != 0) {
		  cout << "LinearSoftmax::backward because gradOutput is nan" << endl;
	  }
	  else {
		  cout << "LinearSoftmax::backward gradOutput normal" << endl;
		  gradOutput.write();
	  }
	  exit(0);
  }*/
  return gradInput;
}

void
LinearSoftmax::updateParameters(float learningRate)
{
  //As in Linear
  weight.gemm(input, 'N', gradOutput, 'T', -learningRate,
      1 - learningRate * weightDecay);
  bias.gemv(gradOutput, 'N', V1col, -learningRate, 1);
}

float
LinearSoftmax::distance2(LinearSoftmax& anotherOutput) {
	floatTensor distMatrix;
	distMatrix.copy(this->weight);
	distMatrix.axpy(anotherOutput.weight, -1);
	float res1 = distMatrix.sumSquared();
	distMatrix.resize(this->bias);
	distMatrix.copy(this->bias);
	distMatrix.axpy(anotherOutput.bias, -1);
	return res1+distMatrix.sumSquared();
}

void
LinearSoftmax::read(ioFile* iof)
{
	// for test
	//cout << "LinearSoftmax::read here" << endl;
  iof->readString(name);
  // for test
  //cout << "LinearSoftmax::read name: " << name << endl;
  weight.read(iof);
  bias.read(iof);
}
void
LinearSoftmax::write(ioFile* iof)
{
	// for test
	cout << "LinearSoftmax::write name: " << name << endl;
	iof->writeString(name);
	weight.write(iof);
	bias.write(iof);
	if (name == "LinearSoftmax_AG") {
		// for test
		cout << "LinearSoftmax::write here" << endl;
		iof->writeFloat(INIT_VALUE_ADAG);
		iof->writeFloat(INIT_VALUE_ADAG);
	}
}
