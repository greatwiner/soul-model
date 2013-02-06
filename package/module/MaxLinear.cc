/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Max Linear Layer, used after LookupTable to have a max function
 at the hidden layer.

 output = element wise max over position (weight_i x R^Tv_i) + bias,
 v_i is the index of the ith context word,
 weight_i is a submatrix of weight for position i


 See Module.H for more detail, for example:
 weight and bias are declared in Module.H
 *******************************************************************/
#include "mainModule.H"

MaxLinear::MaxLinear(int elementNumber, int inputSize, int outputSize,
    int blockSize, outils* otl)
{
  this->blockSize = blockSize;
  weightDecay = 0;
  this->elementNumber = elementNumber;
  this->inputSize = inputSize;
  this->outputSize = outputSize;
  weight.resize(inputSize, elementNumber * outputSize);
  bias.resize(outputSize, 1);
  V1col.resize(blockSize, 1);
  V1col = 1;
  gradInput.resize(elementNumber * inputSize, blockSize);
  internalOutput.resize(elementNumber * outputSize, blockSize);
  internalGradOutput.resize(elementNumber * outputSize, blockSize);
  output.resize(outputSize, blockSize);
  active.resize(outputSize, blockSize);
  copySubInternalOutput.resize(outputSize, blockSize);
  copySubInput.resize(inputSize, blockSize);
  this->otl = otl;
  reset();
}

MaxLinear::~MaxLinear()
{
}

void
MaxLinear::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  V1col.resize(blockSize, 1);
  V1col = 1;
  gradInput.resize(elementNumber * inputSize, blockSize);
  internalOutput.resize(elementNumber * outputSize, blockSize);
  internalGradOutput.resize(elementNumber * outputSize, blockSize);
  output.resize(outputSize, blockSize);
  active.resize(outputSize, blockSize);
  copySubInternalOutput.resize(outputSize, blockSize);
  copySubInput.resize(inputSize, blockSize);
}

void
MaxLinear::reset()
{
  weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  bias.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
}

floatTensor&
MaxLinear::forward(floatTensor& input)
{
  this->input = input;
  output = 0;
  output.ger(bias, V1col, 1);

  int bs;
  int el;
  int os;
  for (el = 0; el < elementNumber; el++)
    {
      subInput.sub(input, el * inputSize, (el + 1) * inputSize - 1, 0,
          blockSize - 1);
      copySubInput.copy(subInput);
      subWeight.sub(weight, 0, inputSize - 1, el * outputSize,
          (el + 1) * outputSize - 1);
      subInternalOutput.sub(internalOutput, el * outputSize,
          (el + 1) * outputSize - 1, 0, blockSize - 1);
      copySubInternalOutput.copy(subInternalOutput);
      copySubInternalOutput.gemm(subWeight, 'T', copySubInput, 'N', 1, 0);
      subInternalOutput.copy(copySubInternalOutput);
    }
  //Max
  float max;
  for (bs = 0; bs < blockSize; bs++)
    {
      for (os = 0; os < outputSize; os++)
        {
          max = -1000000;
          for (el = 0; el < elementNumber; el++)
            {
              if (max < internalOutput(os + el * outputSize, bs))
                {
                  max = internalOutput(os + el * outputSize, bs);
                  active(os, bs) = os + el * outputSize;
                }
            }
          output(os, bs) += max;
        }
    }

  if (infoIof.fo != NULL)
    {
      for (bs = 0; bs < blockSize; bs++)
        {
          for (os = 0; os < outputSize; os++)
            {
              (*infoIof.fo) << (active(os, bs) - os) / outputSize << " ";
            }
          (*infoIof.fo) << endl;
        }
    }
  return output;
}

floatTensor&
MaxLinear::backward(floatTensor& gradOutput)
{

  this->gradOutput = gradOutput;
  //internalGradOutput
  int bs;
  int el;
  int os;
  internalGradOutput = 0;
  for (bs = 0; bs < blockSize; bs++)
    {
      for (os = 0; os < outputSize; os++)
        {
          internalGradOutput(active(os, bs), bs) = gradOutput(os, bs);
        }
    }
  for (el = 0; el < elementNumber; el++)
    {
      subInput.sub(gradInput, el * inputSize, (el + 1) * inputSize - 1, 0,
          blockSize - 1);
      copySubInput.copy(subInput);
      subWeight.sub(weight, 0, inputSize - 1, el * outputSize,
          (el + 1) * outputSize - 1);
      subInternalOutput.sub(internalGradOutput, el * outputSize,
          (el + 1) * outputSize - 1, 0, blockSize - 1);
      copySubInternalOutput.copy(subInternalOutput);
      copySubInput.gemm(subWeight, 'N', copySubInternalOutput, 'N', 1, 0);
      subInput.copy(copySubInput);
    }

  return gradInput;

}

void
MaxLinear::updateParameters(float learningRate)
{

  int el;
  for (el = 0; el < elementNumber; el++)
    {
      subInput.sub(input, el * inputSize, (el + 1) * inputSize - 1, 0,
          blockSize - 1);
      copySubInput.copy(subInput);
      subWeight.sub(weight, 0, inputSize - 1, el * outputSize,
          (el + 1) * outputSize - 1);
      subInternalOutput.sub(internalGradOutput, el * outputSize,
          (el + 1) * outputSize - 1, 0, blockSize - 1);
      copySubInternalOutput.copy(subInternalOutput);
      subWeight.gemm(copySubInput, 'N', copySubInternalOutput, 'T',
          -learningRate, 1 - learningRate * weightDecay);
    }
  bias.gemv(gradOutput, 'N', V1col, -learningRate, 1);

}

void
MaxLinear::read(ioFile* iof)
{
  iof->readString(name);
  weight.read(iof);
  bias.read(iof);
}
void
MaxLinear::write(ioFile* iof)
{
  iof->writeString((char*) "MaxLinear");
  weight.write(iof);
  bias.write(iof);
}
