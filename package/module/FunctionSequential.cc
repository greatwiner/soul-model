#include "mainModule.H"
FunctionSequential::FunctionSequential(int maxSize)
{
  modules = new Module*[maxSize];
  size = 0;
}
FunctionSequential::~FunctionSequential()
{
  for (int idel = 0; idel < size; idel++)
    {
      delete modules[idel];
    }
  delete[] modules;
}
void
FunctionSequential::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  int i;
  for (i = 0; i < size; i++)
    {
      modules[i]->changeBlockSize(blockSize);
    }
  //gradInput = modules[0]->gradInput;
  output = modules[size - 1]->output;

}

void
FunctionSequential::add(Module* module)
{
  size++;
  modules[size - 1] = module;
  output = module->output;
}

floatTensor&
FunctionSequential::forward(floatTensor& input)
{
  int i;
  currentOutput = input;
  for (i = 0; i < size; i++)
    {
      currentOutput = modules[i]->forward(currentOutput);
    }
  return currentOutput;
}

floatTensor&
FunctionSequential::backward(floatTensor& gradOutput)
{
  currentGradOutput = gradOutput;

  Module* currentModule = modules[size - 1];
  Module* previousModule;
  int i;
  for (i = size - 2; i > -1; i--)
    {
      previousModule = modules[i];
      currentGradOutput = currentModule->backward(currentGradOutput);
      currentModule = previousModule;
    }
  currentGradOutput = currentModule->backward(currentGradOutput);
  return currentGradOutput;
}

void
FunctionSequential::updateParameters(float learningRate)
{
  int i;
  for (i = 0; i < size; i++)
    {
      modules[i]->updateParameters(learningRate);
    }
}

void
FunctionSequential::read(ioFile* iof)
{
  int i;
  for (i = 0; i < size; i++)
    {
      modules[i]->read(iof);
    }
}
void
FunctionSequential::write(ioFile* iof)
{
  int i;
  for (i = 0; i < size; i++)
    {
      modules[i]->write(iof);
    }
}

