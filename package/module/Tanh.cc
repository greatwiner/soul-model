#include "mainModule.H"

Tanh::Tanh(int size0, int size1)
{
  blockSize = size1;
  output.resize(size0, size1);
  gradInput.resize(size0, size1);
}

Tanh::~Tanh()
{
}

void
Tanh::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  int size0 = output.size[0];
  int size1 = blockSize;
  output.resize(size0, size1);
  gradInput.resize(size0, size1);
}
floatTensor&
Tanh::forward(floatTensor& input)
{
  output.tanh(input);
  return output;
}

floatTensor&
Tanh::backward(floatTensor& gradOutput)
{
  gradInput.invtanh(output);
  gradInput.product(gradOutput);
  return gradInput;
}
void
Tanh::updateParameters(float learningRate)
{
}
void
Tanh::read(ioFile* iof)
{
  iof->readString(name);
}
void
Tanh::write(ioFile* iof)
{
  iof->writeString((char*) "Tanh");
}
