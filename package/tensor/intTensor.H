/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for integer tensor
 *******************************************************************/


#ifndef INT_TENSOR_INC
#define INT_TENSOR_INC

class intTensor : public Tensor
{
public:
  // Data
  int size[2];
  int stride[2];
  int* data;
  // Method
  intTensor();
  ~intTensor();
  intTensor(int size0, int size1);
  void
  write();
  void
  info();
  // Set, get values
  int&
  operator()(int x);
  int
  operator()(int x) const;

  int&
  operator()(int x, int y);
  int
  operator()(int x, int y) const;

  // Resize size
  int
  resize(int size0, int size1);
  int
  resize(intTensor& src);

  //Manipulate data: Select, sub, copy, tie, transpose

  void
  sub(intTensor& src, int x1, int x2, int y1, int y2);
  void
  select(intTensor& src, int sd, int sliceIndex);
  void
  copy(intTensor& src);
  void
  tieData(intTensor& src);

  void
  t(intTensor& src);
  void
  t();
  // Overload =
  // Important, so don't modify it
  intTensor&
  operator=(int value);
  intTensor&
  operator=(intTensor &src);

  // Read write functions
  void
  read(ioFile* iof);
  // Read without header
  void
  readStrip(ioFile* iof);

  void
  readT(ioFile* iof);

  void
  write(ioFile* iof);
  void
  writeWoSize(ioFile* iof);
  void
  cleanWrite(ioFile* iof);
  // y = a * y
  void
  scal(int alpha);

};

inline int&
intTensor::operator()(int x, int y)
{
  return data[x * stride[0] + y * stride[1]];
}

inline int
intTensor::operator()(int x, int y) const
{
  return data[x * stride[0] + y * stride[1]];
}

inline int&
intTensor::operator()(int x)
{
  return data[x * stride[0]];
}

inline int
intTensor::operator()(int x) const
{
  return data[x * stride[0]];
}

#endif
