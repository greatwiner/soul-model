/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Contains input and output functions for normal, gz or binary files
 *******************************************************************/
#ifndef IOFILE_INC
#define IOFILE_INC

#include "soulConfig.H"

class ioFile
{
public:

  ioFile();
  ~ioFile();
  // format = TEXT; BINARY
  string format;
  // compressed = 1 => .gz, need format = t
  int compressed;
  // Normal files
  ifstream* fi;
  ofstream* fo;
  // Compressed files
  FILE* zipFi;
  FILE* zipFo;
  // Buffer when processing compressed files
  char buf[200];

  // For model files, get the name of this model
  string
  recognition(char* inputFileName);

  // Check if file exists, out means printing a message
  int
  check(char* fileName, int out);
  // get a file to read
  int
  takeReadFile(char* readFileName);
  // get a file to write
  int
  takeWriteFile(char* writeName);
  // close files
  void
  freeWriteFile();
  void
  freeReadFile();
  // return true for end of file
  int
  getEOF();
  // Buffer for read, write
  char readCharBuff[200];

  // Marco define read, write function for int, long ... one and array
#define HHRW(Type, type)\
  int read##Type(type& tensor##Type);\
  int write##Type(const type tensor##Type);\
  type* read##Type##Array(type* tensor##Type, int n);\
  int write##Type##Array(const type* tensor##Type, int n)\

  HHRW(Int, int);HHRW(Long, long);HHRW(Float, float);HHRW(Char, char);

  // Read, write strings
  int
  writeNorString(const string str);

  int
  writeString(const string str);

  int
  readString(string& out);

  int
  getLine(string& out);

};

#endif

