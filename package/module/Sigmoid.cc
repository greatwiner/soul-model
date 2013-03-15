#include "mainModule.H"

Sigmoid::Sigmoid(int size0, int size1)
{
  blockSize = size1;
  output.resize(size0, size1);
  gradInput.resize(size0, size1);
}

Sigmoid::~Sigmoid()
{
}

void
Sigmoid::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  int size0 = output.size[0];
  int size1 = blockSize;
  output.resize(size0, size1);
  gradInput.resize(size0, size1);
}

floatTensor&
Sigmoid::forward(floatTensor& input)
{
  output.sigm(input);

  if (infoIof.fo != NULL)
    {
      int bs;
      int os;
      int outputSize = output.size[0];
      for (bs = 0; bs < blockSize; bs++)
        {
          for (os = 0; os < outputSize; os++)
            {
              infoIof.writeFloat(output(os, bs));
            }
        }
    }
  return output;
}

floatTensor&
Sigmoid::backward(floatTensor& gradOutput)
{
  gradInput.invsigm(output);
  gradInput.product(gradOutput);
  return gradInput;
}
void
Sigmoid::updateParameters(float learningRate)
{
}
void
Sigmoid::read(ioFile* iof)
{
  iof->readString(name);
}
void
Sigmoid::write(ioFile* iof)
{
  iof->writeString((char*) "Sigmoid");
}

