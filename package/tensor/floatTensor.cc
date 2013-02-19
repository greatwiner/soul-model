#include "mainTensor.H"
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

#if (PROC == INTEL)

#define copy_ scopy
#define exp_  vsExp
#define log_  vsLn
#define scal_ sscal
#define ger_  sger
#define dot_  sdot
#define gemv_ sgemv
#define gemm_ sgemm
#define axpy_ saxpy
#define pow_  vsPowx
#define tanh_ vsTanh

#elif (PROC == AMD)

#define copy_ scopy_
#define exp_  vrsa_expf
#define log_  vrsa_logf
#define scal_ sscal_
#define ger_  sger_
#define dot_  sdot_
#define gemv_ sgemv_
#define gemm_ sgemm_
#define axpy_ saxpy_
#define pow_  vrsa_powxf
#define tanh_ lost_

#elif (PROC == CBLAS)

#define copy_ scopy_
#define exp_  lost_
#define log_  lost_
#define scal_ sscal_
#define ger_  sger_
#define dot_  sdotsub_
#define gemv_ sgemv_
#define gemm_ sgemm_
#define axpy_ saxpy_
#define pow_  lost_
#define tanh_ lost_

#elif (PROC == OPENBLAS)

#define copy_ scopy_
#define exp_  lost_
#define log_  lost_
#define scal_ sscal_
#define ger_  sger_
#define dot_  sdot_
#define gemv_ sgemv_
#define gemm_ sgemm_
#define axpy_ saxpy_
#define pow_  lost_
#define tanh_ lost_

#endif

floatTensor::floatTensor()
{
// For Mkl, low accuracy for VML is faster
#if (PROC == INTEL)
  vmlSetMode(VML_LA | VML_FLOAT_CONSISTENT);
#endif
  size = NULL;
  stride = NULL;
  data = NULL;
  haveMemory = 0;
  length = 0;
}

floatTensor&
floatTensor::operator =(floatTensor& src)
{
  size = src.size;
  stride = src.stride;
  data = src.data;
  haveMemory = -1;
  length = src.length;
  return *this;
}

floatTensor::~floatTensor()
{
  if (haveMemory != -1)
    {
      if (size != NULL)
        {
          delete[] size;
        }
      if (stride != NULL)
        {
          delete[] stride;
        }
      if (data != NULL && haveMemory)
        {
          delete[] data;
        }
    }
}

floatTensor::floatTensor(int size0, int size1)
{
  size = new int[2];
  size[0] = size0;
  size[1] = size1;
  stride = new int[2];
  stride[0] = 1;
  stride[1] = size0;
  try
    {
      data = new float[size0 * size1];
    }
  catch (bad_alloc& ba)
    {
      cerr << "ERROR floatTensor resize " << ba.what() << endl;
    }

  haveMemory = 1;
  length = size[0] * size[1];
}

void
floatTensor::write()
{
  cout << "# floatTensor" << endl;
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
floatTensor::info()
{
  cout << "# floatTensor" << endl;
  cout << "dimension " << size[0] << " " << size[1] << endl;
  cout << "stride " << stride[0] << " " << stride[1] << endl;
}
int
floatTensor::resize(int size0, int size1)
{
  if (haveMemory == 1 && size[0] == size0 && size[1] == size1)
    {
      return 0;
    }
  if (size == NULL)
    {
      size = new int[2];
      stride = new int[2];
    }
  else
    {
      delete[] data;
    }
  size[0] = size0;
  size[1] = size1;
  stride[0] = 1;
  stride[1] = size0;
  try
    {
      data = new float[size0 * size1];
    }
  catch (bad_alloc& ba)
    {
      cerr << "ERROR floatTensor resize " << ba.what() << endl;
    }

  haveMemory = 1;
  length = size[0] * size[1];
  return 1;
}

int
floatTensor::resize(floatTensor& src)
{
  if (haveMemory == 1 && size[0] == src.size[0] && size[1] == src.size[1])
    {
      return 0;
    }
  if (size == NULL)
    {
      size = new int[2];
      stride = new int[2];
    }
  else
    {
      delete[] data;
    }
  size[0] = src.size[0];
  size[1] = src.size[1];
  stride[0] = src.stride[0];
  stride[1] = src.stride[1];
  try
    {
      data = new float[size[0] * size[1]];
    }
  catch (bad_alloc& ba)
    {
      cerr << "ERROR floatTensor resize " << ba.what() << endl;
    }

  haveMemory = 1;
  length = size[0] * size[1];
  return 1;
}
void
floatTensor::sub(floatTensor& src, int x1, int x2, int y1, int y2)
{
  haveMemory = 0;
  if (size == NULL)
    {
      size = new int[2];
      stride = new int[2];
    }
  size[0] = x2 - x1 + 1;
  size[1] = y2 - y1 + 1;
  stride[0] = src.stride[0];
  stride[1] = src.stride[1];
  data = &(src(x1, y1));
  length = size[0] * size[1];
}
void
floatTensor::select(floatTensor& src, int sd, int sliceIndex)
{
  haveMemory = 0;
  if (size == NULL)
    {
      size = new int[2];
      stride = new int[2];
    }
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
floatTensor&
floatTensor::operator=(float value)
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
floatTensor::copy(floatTensor& src)
{
  if (size == NULL)
    {
      resize(src);
    }
  if (length != src.length || size[0] != src.size[0] || size[1] != src.size[1])
    {
      cerr << "ERROR: Copy float tensor with different size\n";
      cerr << length << " " << src.length << " " << size[0] << " "
          << src.size[0] << " " << size[1] << " " << src.size[1] << endl;
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
floatTensor::tieData(floatTensor& src)
{
  haveMemory = 0;
  delete[] data;
  if (length != src.length || size[0] != src.size[0] || size[1] != src.size[1])
    {
      cerr << "ERROR: Tie tensor with different size\n";
      exit(1);
    }
  data = src.data;
}

void
floatTensor::mexp(floatTensor& src)
{
#if (PROC == INTEL) || (PROC == AMD)
  exp_(length, src.data, data); // y = exp(x)
#else
  for (int i = 0; i < length; i++)
    {
      data[i] = exp(src.data[i]);
    }
#endif
}

void
floatTensor::mlog(floatTensor& src)
{
#if (PROC == INTEL) || (PROC == AMD)
  log_(length, src.data, data); // y = log(x)
#else
  for (int i = 0; i < length; i++)
    {
      data[i] = log(src.data[i]);
    }
#endif
}

void
floatTensor::sigm(floatTensor& src)
{
  copy(src);
  scal(-1.0); // y = -x
  mexp(*this); // y = exp(-x)
  for (int i = 0; i < length; i++)
    {
      data[i] = 1.0 / (1.0 + data[i]);
    }
}

void
floatTensor::tanh(floatTensor& src)
{
#if (PROC == INTEL)
  tanh_(length, src.data, data);
#else
  int n = 1;
  //copy_(&length, src.data, &n, data, &n); //copy x to y
  copy(src);
  scal(2.0); // y = 2x
  mexp(*this); // y = exp(2x)
  for (int i = 0; i < length; i++)
    {
      data[i] = (data[i] - 1.0) / (data[i] + 1.0);
    }
#endif
}

void
floatTensor::invsigm(floatTensor& src)
{
#if (PROC == INTEL) || (PROC == AMD)
  pow_(length, src.data, 2, data);// y = x * x
#else
  for (int i = 0; i < length; i++)
    {
      data[i] = src.data[i] * src.data[i];
    }
#endif
  scal(-1.0); // y = - x * x
  axpy(src, 1.0);
}

void
floatTensor::invtanh(floatTensor& src)
{
#if (PROC == INTEL) || (PROC == AMD)
  pow_(length, src.data, 2, data);// y = x * x
#else
  for (int i = 0; i < length; i++)
    {
      data[i] = src.data[i] * src.data[i];
    }
#endif
  for (int i = 0; i < length; i++)
    {
      data[i] = 1.0 - data[i];
    }
}

void
floatTensor::softmax(floatTensor& src, floatTensor& vCol, floatTensor& v1row)
{
  for (int i = 0; i < size[1]; i++)
    {
      float min = 10000000;
      for (int j = 0; j < size[0]; j++)
        {
          if (src(j, i) < min)
            {
              min = src(j, i);
            }
        }
      for (int j = 0; j < size[0]; j++)
        {
          src(j, i) -= min;
        }
    }
  mexp(src);
  vCol.gemv(*this, 'T', v1row, 1, 0);
  floatTensor selectThis;
  for (int i = 0; i < size[1]; i++)
    {
      selectThis.select(*this, 1, i);
      selectThis.scal(1.0 / vCol(i));
    }
}

void
floatTensor::product(floatTensor& src)
{
  for (int i = 0; i < size[0]; i++)
    {
      for (int j = 0; j < size[1]; j++)
        {
          data[i * stride[0] + j * stride[1]] *= src.data[i * stride[0] + j
              * stride[1]];
        }
    }
}
void
floatTensor::scal(float alpha)//y = ay
{
  int n = 1;
  scal_(&length, &alpha, data, &n);
}

void
floatTensor::axpy(floatTensor& tensor1, float alpha)//y = ax + y
{
  int lstride = 1;
  axpy_(&length, &alpha, tensor1.data, &lstride, data, &lstride);
}

void
floatTensor::ger(floatTensor& x, floatTensor& y, float alpha) // A =axyT + A
{
  if (stride[0] == 1)
    {
      ger_(&x.size[0], &y.size[0], &alpha, x.data, &x.stride[0], y.data,
          &y.stride[0], data, &stride[1]);
    }
  else
    {
      ger_(&y.size[0], &x.size[0], &alpha, y.data, &y.stride[0], x.data,
          &x.stride[0], data, &stride[0]);
    }
}

void
floatTensor::gemv(floatTensor& M, char transM, floatTensor& v, float alpha,
    float beta)//y = aAx + by
{
  int ldm;
  ldm = M.size[0];

#if (PROC == INTEL) || (PROC == CBLAS) || (PROC == OPENBLAS)
  gemv_(&transM, &M.size[0], &M.size[1], &alpha, M.data, &ldm, v.data,
      &v.stride[0], &beta, data, &stride[0]);
#elif (PROC == AMD)
  gemv_(&transM, &M.size[0], &M.size[1], &alpha, M.data, &ldm, v.data,
      &v.stride[0], &beta, data, &stride[0], 1);
#endif
}

void
floatTensor::gemm(floatTensor& A, char transA, floatTensor& B, char transB,
    float alpha, float beta) // C := aAB + bC
{
  int m, n, k;
  int lda, ldb, ldc;
  m = size[0];
  n = size[1];
  ldc = m;
  if (transA == 'N')
    {
      k = A.size[1];
      lda = m;
    }
  else
    {
      k = A.size[0];
      lda = k;
    }
  if (transB == 'N')
    {
      ldb = k;
    }
  else
    {
      ldb = n;
    }

#if (PROC == INTEL) || (PROC == CBLAS) || (PROC == OPENBLAS) 
  gemm_(&transA, &transB, &m, &n, &k, &alpha, A.data, &lda, B.data, &ldb,
      &beta, data, &ldc);
#elif (PROC == AMD)
  gemm_(&transA, &transB, &m, &n, &k, &alpha, A.data, &lda, B.data, &ldb,
      &beta, data, &ldc, 1, 1);
#endif
}

float
floatTensor::dot(floatTensor& src)
{
  if (stride[0] == src.stride[0] && stride[1] == src.stride[1])
    {
      int n = 1;
#if (PROC == INTEL) || (PROC == AMD) || (PROC == OPENBLAS)
      return dot_(&length, src.data, &n, data, &n);
#elif (PROC == CBLAS)
      float dot;
      dot_(&length, src.data, &n, data, &n, &dot);
      return dot;
#endif
    }
  else
    {
      float sum = 0;
      for (int i = 0; i < size[0]; i++)
        {
          for (int j = 0; j < size[1]; j++)
            {
              sum += data[i * stride[0] + j * stride[1]] * src.data[i
                  * stride[0] + j * stride[1]];
            }
        }
      return sum;
    }

}
void
floatTensor::read(ioFile* iof)
{
  int* newSize = new int[2];
  newSize = iof->readIntArray(newSize, 2);
  resize(newSize[0], newSize[1]);
  delete[] newSize;

  length = size[0] * size[1];
  data = iof->readFloatArray(data, length);
}
void
floatTensor::write(ioFile* iof)
{
  if (haveMemory == 0)
    {
      cerr << "ERROR: Try writing tensor with haveMemory = 0" << endl;
    }
  iof->writeIntArray(size, 2);
  iof->writeFloatArray(data, length);
}

void
floatTensor::uniform(float a, float b, outils* otl)
{
  for (int i = 0; i < size[0]; i++)
    {
      for (int j = 0; j < size[1]; j++)
        {
          data[i * stride[0] + j * stride[1]] = a + otl->genrand() * (b - a);
        }
    }

}

void
floatTensor::writeWoSize(ioFile* iof)
{
  if (haveMemory == 0)
    {
      cerr << "ERROR: Try writing tensor with haveMemory = 0" << endl;
    }
  iof->writeFloatArray(data, length);

}

void
floatTensor::readT(ioFile* iof)
{
  int* newSize = new int[2];
  newSize = iof->readIntArray(newSize, 2);
  resize(newSize[0], newSize[1]);
  length = size[0] * size[1];
  delete[] newSize;
  for (int i = 0; i < size[0]; i++)
    {
      for (int j = 0; j < size[1]; j++)
        {
          iof->readFloat(data[i * stride[0] + j * stride[1]]);
        }
    }
}

void
floatTensor::readStrip(ioFile* iof)
{
  for (int i = 0; i < size[0]; i++)
    {
      for (int j = 0; j < size[1]; j++)
        {
          iof->readFloat(data[i * stride[0] + j * stride[1]]);
        }
    }
}

void
floatTensor::initializeNormal() {
	for (int i = 0; i < this->size[0]; i ++) {
		for (int j = 0; j < this->size[1]; j ++) {
			data[i*stride[0] + j*stride[1]] = initializeNormalOneElement();
		}
	}
}

void
floatTensor::log(floatTensor& src)
{
	log_(length, src.data, data); // y = exp(-x)
}

float
floatTensor::initializeNormalOneElement() {
	/*// uniformly distributed float numbers
	float U1=((float)rand()/(float)RAND_MAX);
	float U2=((float)rand()/(float)RAND_MAX);
	return sqrt(-2*::log(U1))*cos(2*M_PI*U2);*/
	return 0;
}

float
floatTensor::sumSquared() {
	return this->dot(*this);
}
