/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Main header to include all header files for tensor.
 Tensor is the variables for vector and matrix. Vector is considered
 as one matrix with 1 as the second dimension.
 outils is used for random initialization
 Data are considered in fortran mode (column major order)
 *******************************************************************/

#define INTEL    0
#define AMD      1
#define CBLAS    2
#define OPENBLAS 3


#if (PROC == INTEL)

#include "mkl_blas.h"
#include "mkl_vml.h"

#elif (PROC == AMD)

#include "acml.h"
#include "acml_mv.h"

#elif (PROC == CBLAS)

#include "cblas_f77.h"

#elif (PROC == OPENBLAS)

#include "f77blas.h"

#endif

#include "ioFile.H"
#include "outils.H"
#include "Tensor.H"
#include "floatTensor.H"
#include "intTensor.H"

