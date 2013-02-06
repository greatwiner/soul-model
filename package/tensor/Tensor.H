/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Main class for Tensor
 Data are considered in fortran mode (column major order)
 *******************************************************************/

#ifndef TENSOR_INC
#define TENSOR_INC

class Tensor
{
public:
  //Data
  // haveMemory
  //  1: Free all: data, size, stride,
  //  0: Don't need to free data,
  // -1: Don't free anything
  int haveMemory;
  // Total memory size for data
  int length;
  Tensor();
  ~Tensor();

};

#endif
