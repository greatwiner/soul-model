#include "mainTensor.H"

intTensor::intTensor()
{
  size[0] = -1;
  stride[0] = -1;
  data = NULL;
  haveMemory = 0;
  length = 0;
}

intTensor&
intTensor::operator =(intTensor& src)
{
  memcpy(size, src.size, 2 * sizeof(int));
  memcpy(stride, src.stride, 2 * sizeof(int));
  data = src.data;
  haveMemory = -1;
  length = src.length;
  return *this;
}

intTensor::~intTensor()
{
  if (data != NULL && (haveMemory == 1))
    {
      delete[] data;
    }
}

intTensor::intTensor(int size0, int size1)
{
  size[0] = size0;
  size[1] = size1;
  stride[0] = 1;
  stride[1] = size0;
  try
    {
      data = new int[size0 * size1];
    }
  catch (bad_alloc& ba)
    {
      cerr << "ERROR intTensor resize " << ba.what() << endl;
    }

  haveMemory = 1;
  length = size[0] * size[1];
}

void
intTensor::write()
{
  cout << "# intTensor" << endl;
  cout << "dimension " << size[0] << " " << size[1] << endl;
  for (int i = 0; i < size[0]; i++)
    {
      for (int j = 0; j < size[1]; j++)
        {
          cout << data[i * stride[0] + j * stride[1]] << " ";
        }
      cout << endl;
    }
}

void
intTensor::info()
{
  cout << "# intTensor" << endl;
  cout << "dimension " << size[0] << " " << size[1] << endl;
}

int
intTensor::resize(int size0, int size1)
{
  if (haveMemory == 1 && size[0] == size0 && size[1] == size1)
    {
      return 0;
    }
  if (data != NULL && haveMemory == 1)
    {
      delete[] data;
    }

  size[0] = size0;
  size[1] = size1;
  stride[0] = 1;
  stride[1] = size0;
  try
    {
      data = new int[size0 * size1];
    }
  catch (bad_alloc& ba)
    {
      cerr << "ERROR intTensor resize " << ba.what() << endl;
    }
  haveMemory = 1;
  length = size[0] * size[1];
  return 1;
}

int
intTensor::resize(intTensor& src)
{
  return resize(src.size[0], src.size[1]);
}

void
intTensor::sub(intTensor& src, int x1, int x2, int y1, int y2)
{
  if (data != NULL && haveMemory == 1)
    {
      delete[] data;
      cerr << "WARNING: sub with tensor haveMemory = 1" << endl;
    }
  haveMemory = 0;
  size[0] = x2 - x1 + 1;
  size[1] = y2 - y1 + 1;
  stride[0] = src.stride[0];
  stride[1] = src.stride[1];
  data = &(src(x1, y1));
  length = size[0] * size[1];
}
void
intTensor::select(intTensor& src, int sd, int sliceIndex)
{
  if (data != NULL && haveMemory == 1)
    {
      delete[] data;
      cerr << "WARNING: sub with tensor haveMemory = 1" << endl;
    }
  haveMemory = 0;
  size[0] = src.size[1 - sd];
  size[1] = 1;
  stride[0] = src.stride[1 - sd];
  stride[1] = size[0];
  if (sd)
    {
      data = &(src(0, sliceIndex));
    }
  else
    {
      data = &(src(sliceIndex, 0));
    }
  length = size[0];
}
//Overload =
intTensor&
intTensor::operator=(int value)
{
  for (int i = 0; i < size[0]; i++)
    {
      for (int j = 0; j < size[1]; j++)
        {
          data[i * stride[0] + j * stride[1]] = value;
        }
    }
  return *this;
}

void
intTensor::copy(intTensor& src)
{
  if (size[0] == -1)
    {
      resize(src);
    }
  if (length != src.length || size[0] != src.size[0] || size[1] != src.size[1])
    {
      cerr << "ERROR: Copy int tensor with different size\n";
      exit(1);
    }
  for (int i = 0; i < size[0]; i++)
    {
      for (int j = 0; j < size[1]; j++)
        {
          data[i * stride[0] + j * stride[1]] = src.data[i * src.stride[0] + j
              * src.stride[1]];
        }
    }
}

void
intTensor::tieData(intTensor& src)
{
  if (haveMemory == 1 && data != NULL)
    {
      delete[] data;
    }
  haveMemory = 0;

  if (length != src.length || size[0] != src.size[0] || size[1] != src.size[1])
    {
      cerr << "ERROR: Tie tensor with different size\n";
      exit(1);
    }
  data = src.data;
}

void
intTensor::scal(int alpha) //y = ay
{
  for (int i = 0; i < length; i++)
    {
      data[i] = data[i] * alpha;
    }
}

void
intTensor::t(intTensor &src)
{
  haveMemory = 0;
  size[0] = src.size[1];
  size[1] = src.size[0];
  stride[0] = src.stride[1];
  stride[1] = src.stride[0];
  data = src.data;
  length = src.length;
}

void
intTensor::t()
{
  int m;
  m = size[0];
  size[0] = size[1];
  size[1] = m;
  m = stride[0];
  stride[0] = stride[1];
  stride[1] = m;
}

void
intTensor::read(ioFile* iof)
{

  int* newSize = new int[2];
  newSize = iof->readIntArray(newSize, 2);
  resize(newSize[0], newSize[1]);
  delete[] newSize;

  length = size[0] * size[1];
  data = iof->readIntArray(data, length);
}

void
intTensor::readStrip(ioFile* iof)
{
  for (int i = 0; i < size[0]; i++)
    {
      for (int j = 0; j < size[1]; j++)
        {
          iof->readInt(data[i * stride[0] + j * stride[1]]);
        }
    }
}

void
intTensor::write(ioFile* iof)
{
  if (haveMemory == 0)
    {
      cerr << "ERROR: Try writing tensor with haveMemory = 0" << endl;
    }
  iof->writeIntArray(size, 2);
  iof->writeIntArray(data, length);
}

void
intTensor::writeWoSize(ioFile* iof)
{
  if (haveMemory == 0)
    {
      cerr << "ERROR: Try writing tensor with haveMemory = 0" << endl;
    }
  iof->writeIntArray(data, length);

}

void
intTensor::readT(ioFile* iof)
{
  int* newSize = new int[2];
  newSize = iof->readIntArray(newSize, 2);
  resize(newSize[0], newSize[1]);
  delete[] newSize;
  length = size[0] * size[1];
  for (int i = 0; i < size[0]; i++)
    {
      for (int j = 0; j < size[1]; j++)
        {
          iof->readInt(data[i * stride[0] + j * stride[1]]);
        }
    }
}
