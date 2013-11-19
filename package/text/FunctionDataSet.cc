#include "text.H"

FunctionDataSet::~FunctionDataSet()
{
  if (data != NULL)
    {
      delete[] data;
    }
}

FunctionDataSet::FunctionDataSet(int dim, int classNumber)
{
  this->dim = dim;
  this->classNumber = classNumber;
  data = NULL;
  dataNumber = 0;
  dataTensor.resize(1, 1);

  try
    {
      data = new float[FUNC_NGRAM_NUMBER * (dim + 1)];
    }
  catch (bad_alloc& ba)
    {
      cerr << "FunctionDataSet bad_alloc caught: " << ba.what() << endl;
      exit(1);
    }

}

int
FunctionDataSet::checkBlankString(string line)
{
  for (int i = 0; i < line.length(); i++)
    {
      if (line[i] != ' ')
        {
          return 0;
        }
    }
  return 1;
}

int
FunctionDataSet::addLine(string line)
{
  istringstream streamLine(line);
  float value;
  int i = 0;
  while (streamLine >> value)
    {
      data[dataNumber * (dim + 1) + i] = value;
      i = i + 1;
    }
  dataNumber++;
  return 1;
}

int
FunctionDataSet::readText(ioFile* iof)
{
  int readLineNumber = 0;
  string line;
  while (!iof->getEOF())
    {
      if (iof->getLine(line))
        {
          if (!checkBlankString(line))
            {
              addLine(line);
            }
        }
      readLineNumber++;
#if PRINT_DEBUG
      if (readLineNumber % NLINEPRINT == 0)
        {
          cout << readLineNumber << " ... " << flush;
        }
#endif
    }
#if PRINT_DEBUG
  cout << endl;
#endif
  if (dataNumber > FUNC_NGRAM_NUMBER)
    {
      cerr << "Not enough memory" << endl;
    }
  return 1;
}

int
FunctionDataSet::readBiNgram(ioFile* iof)
{
  int i;
  int readDataNumber;
  int readLineNumber = 0;
  iof->readInt(readDataNumber);
  int N;
  iof->readInt(N);
  if (N != dim + 1)
    {
      cerr << "dim in data is wrong" << endl;
      return 0;
    }

  float readTextNgram[N];
  while (!iof->getEOF())
    {
      iof->readFloatArray(readTextNgram, dim + 1);
      if (iof->getEOF())
        {
          break;
        }
      for (i = 0; i < dim + 1; i++)
        {
          data[dataNumber * (dim + 1) + i] = readTextNgram[i];
        }
      dataNumber++;
      readLineNumber++;
#if PRINT_DEBUG
      if (readLineNumber % NLINEPRINT == 0)
        {
          cout << readLineNumber << " ... " << flush;
        }
#endif
    }
#if PRINT_DEBUG
  cout << endl;
#endif
  return dataNumber;
}

int
FunctionDataSet::readAllClassBiNgram(ioFile* iof)
{
  int i;
  int readDataNumber;
  int readLineNumber = 0;
  iof->readInt(readDataNumber);
  int N;
  iof->readInt(N);
  if (N != dim + 1)
    {
      cerr << "dim in data is wrong" << endl;
      return 0;
    }

  float readTextNgram[N];
  int iclass;
  while (!iof->getEOF())
    {
      iof->readFloatArray(readTextNgram, dim + 1);
      if (iof->getEOF())
        {
          break;
        }
      for (iclass = 0; iclass < classNumber; iclass++)
        {
          for (i = 0; i < dim; i++)
            {
              data[dataNumber * (dim + 1) + i] = readTextNgram[i];
            }
          data[dataNumber * (dim + 1) + dim] = (float) iclass;
          dataNumber++;
        }
      readLineNumber++;
#if PRINT_DEBUG
      if (readLineNumber % NLINEPRINT == 0)
        {
          cout << readLineNumber << " ... " << flush;
        }
#endif
    }
#if PRINT_DEBUG
  cout << endl;
#endif
  return dataNumber;
}

void
FunctionDataSet::createTensor()
{
  dataTensor.haveMemory = 0;
  dataTensor.setSize(0, dataNumber);
  //dataTensor.size[0] = dataNumber;
  dataTensor.setSize(1, dim+1);
  //dataTensor.size[1] = dim + 1;
  dataTensor.stride[0] = dim + 1;
  dataTensor.stride[1] = 1;
  if (dataTensor.data != data)
    {
      delete[] dataTensor.data;
      dataTensor.data = data;
    }
  probTensor.resize(dataNumber, 1);
}

void
FunctionDataSet::shuffle(int times)
{
  int dim1 = dim + 1;
  float *tg = new float[dim1 * sizeof(float)];
  int i;
  int p1, p2;
  for (i = 0; i < times * dataNumber; i++)
    {
      p1 = (int) (dataNumber * drand48());
      p2 = (int) (dataNumber * drand48());
      memcpy(tg, data + p1 * dim1, dim1 * sizeof(float));
      memcpy(data + p1 * dim1, data + p2 * dim1, dim1 * sizeof(float));
      memcpy(data + p2 * dim1, tg, dim1 * sizeof(float));
    }
}

int
FunctionDataSet::writeNgram(ioFile* iof)
{
  iof->writeInt(dataNumber);
  iof->writeInt(dim + 1);
  int dataId = 0;
  for (dataId = 0; dataId < dataNumber; dataId++)
    {
      iof->writeFloatArray(data + dataId * (dim + 1), dim + 1);
    }
  return dataNumber;
}
float
FunctionDataSet::computePerplexity()
{
  float perplexity = 0;
  for (int i = 0; i < probTensor.length; i++)
    {
      perplexity += log(probTensor(i));
    }
  perplexity = exp(-perplexity / dataNumber);
  return perplexity;
}
