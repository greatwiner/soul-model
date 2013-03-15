/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Tensor is vector or matrix for float data with single precision
 If tensor is vector (size M), we consider it as one matrix of M x 1
 BLAS (not CBLAS) functions are used for efficient calculation
 Data are considered in fortran mode (column major order)
 *******************************************************************/

#ifndef FLOAT_TENSOR_INC
#define FLOAT_TENSOR_INC

class floatTensor : public Tensor
{
public:
  // Data
  int* size;
  int* stride;
  float* data;
  // Method
  floatTensor();
  ~floatTensor();
  floatTensor(int size0, int size1);
  // Write to stdout
  void
  write();
  // Write information to stdout
  void
  info();
  // Set, get values
  float&
  operator()(int x);
  float
  operator()(int x) const;

  float&
  operator()(int x, int y);
  float
  operator()(int x, int y) const;

  // Resize size
  int
  resize(int size0, int size1);
  int
  resize(floatTensor& src);

  // Manipulate data
  // Select, sub, copy, tie
  void
  sub(floatTensor& src, int x1, int x2, int y1, int y2);
  void
  select(floatTensor& src, int sd, int sliceIndex);
  void
  copy(floatTensor& src);
  void
  tieData(floatTensor& src);

  // Overload =
  // Important, so don't modify
  floatTensor&
  operator=(float value);
  floatTensor&
  operator=(floatTensor &src);

  // Math simple function like sigm, tanh
  // invsigm, invtanh are backward function of sigm, tanh in neural network
  // Two tensors have the same dimension, size, stride and at least one stride = 1
  void
  mexp(floatTensor& src);
  void
  mlog(floatTensor& src);

  void
  sigm(floatTensor& src);
  void
  tanh(floatTensor& src);
  void
  invsigm(floatTensor& src);
  void
  invtanh(floatTensor& src);
  void
  softmax(floatTensor& src, floatTensor &vCol, floatTensor &v1row);

  // y = x * y (element wise)
  void
  product(floatTensor& src);
  // Math matrix vector BLAS
  // y = ay
  void
  scal(float alpha);
  // y = ax + y
  void
  axpy(floatTensor& tensor1, float alpha);
  // A =axyT + A
  void
  ger(floatTensor& x, floatTensor& y, float alpha);
  // y = aAx + by
  void
  gemv(floatTensor& M, char transM, floatTensor& v, float alpha, float beta);
  // C = aAB + bC
  void
  gemm(floatTensor& A, char transA, floatTensor& B, char transB, float alpha,
      float beta);
  // v = x * y
  float
  dot(floatTensor& src);

  float
  sumSquared();

  float
  averageSquare();

  float
  initializeNormalOneElement();

  void
  log(floatTensor& src);

  void
  initializeNormal();

  // Read write function,
  // when haveMemory = 0, we don't write it, so when read it,
  // we must use the same code to create the pointer data.
  // e.g. lbl model, weightLinear.t(weigthLookuptable):
  // we save only weigthLookuptable in reality.

  // Read in default order (column major order)
  // In file: header: 2 4, then data: 1 2 3 4 5 6 7 8
  // tensor is a matrix 2 x 4, we will have
  // tensor.data = [1, 2, 3, 4, 5, 6, 7, 8] (as in file)
  // it represents a matrix:
  // 1 3 5 7
  // 2 4 6 8
  void
  read(ioFile* iof);

  //Read in row major order
  // In file: header: 2 4, then data: 1 2 3 4 5 6 7 8
  // tensor is a matrix 2 x 4, we will have
  // tensor.data = [1, 5, 2, 6, 3, 7, 4, 8]
  // it represents a matrix:
  // 1 2 3 4
  // 5 6 7 8
  void
  readT(ioFile* iof);

  // Read in row major order but without header
  // In file 1 2 3 4 5 6 7 8
  // tensor is pre-defined as a matrix 2 x 4,
  // tensor.data = [1, 5, 2, 6, 3, 7, 4, 8]
  // after reading, it represents a matrix:
  // 1 2 3 4
  // 5 6 7 8
  void
  readStrip(ioFile* iof);

  void
  write(ioFile* iof);
  void
  writeWoSize(ioFile* iof);

  // Sample from uniform distribution
  void
  uniform(float a, float b, outils* otl);

};

inline float&
floatTensor::operator()(int x, int y)
{
  return data[x * stride[0] + y * stride[1]];
}

inline float
floatTensor::operator()(int x, int y) const
{
  return data[x * stride[0] + y * stride[1]];
}

inline float&
floatTensor::operator()(int x)
{
  return data[x * stride[0]];
}

inline float
floatTensor::operator()(int x) const
{
  return data[x * stride[0]];
}

#endif
