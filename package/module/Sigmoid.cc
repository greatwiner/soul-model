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
	// for test
	//cout << "Sigmoid::forward here" << endl;
	//cout << "Sigmoid::forward input: " << endl;
	//input.info();
	//cout << "Sigmoid::forward output: " << endl;
	//output.info();
  output.sigm(input);
  // for test
  //cout << "Sigmoid::forward here1" << endl;

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
  /*if (gradInput.testNan() != 0) {
	  cout << "Sigmoid::backward gradInput is nan" << endl;
	  cout << "Sigmoid::backward program will exit" << endl;
	  if (gradOutput.testNan() != 0) {
		  cout << "Sigmoid::backward because gradOutput is nan" << endl;
	  }
	  if (output.testNan() != 0) {
		  cout << "Sigmoid::backward because output is nan" << endl;
	  }
	  exit(0);
  }*/
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

