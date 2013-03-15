/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Specific class for language model. This layer aims to be the softmax
 layer which takes as input the last hidden layer to predict
 word (class) probabilities.
 *******************************************************************/
#include "mainModule.H"
LinearSoftmax::LinearSoftmax(int inputSize, int outputSize, int blockSize,
    outils* otl)
{
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

  // For each column, minus minimum value
  for (int i = 0; i < output.size[1]; i++)
    {
      float min = 10000000;
      for (int j = 0; j < output.size[0]; j++)
        {
          if (preOutput(j, i) < min)
            {
              min = preOutput(j, i);
            }
        }
      for (int j = 0; j < output.size[0]; j++)
        {
          preOutput(j, i) -= min;
        }
    }
  // output = e^(preOutput)
  output.mexp(preOutput);

  // softmaxVCol contains the sum for each column
  softmaxVCol.gemv(output, 'T', softmaxV1row, 1, 0);

  // For each column, divide by the sum to have
  // for each element e^x_i / \sum_j e^x_j
  for (int i = 0; i < output.size[1]; i++)
    {
      selectOutput.select(output, 1, i);
      selectOutput.scal(1.0 / softmaxVCol(i));
    }

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
  gradOutput.copy(output);
  int i;
  this->input = input;
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
          selectGradOutput.select(gradOutput, 1, i);
          selectGradOutput = 0;
        }
    }
  gradInput.gemm(weight, 'N', gradOutput, 'N', 1, 0);
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

void
LinearSoftmax::read(ioFile* iof)
{
  iof->readString(name);
  weight.read(iof);
  bias.read(iof);
}
void
LinearSoftmax::write(ioFile* iof)
{
  iof->writeString((char*) "LinearSoftmax");
  weight.write(iof);
  bias.write(iof);
}
