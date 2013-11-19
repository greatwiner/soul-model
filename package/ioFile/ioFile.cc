/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Contains input and output functions for normal, gz or binary files
 *******************************************************************/

#include "ioFile.H"

ioFile::ioFile()
{
  fi = NULL;
  fo = NULL;
  zipFi = NULL;
  zipFo = NULL;
  format = BINARY;
  compressed = 0;
}
ioFile::~ioFile()
{
  if (fi != NULL)
    {
      fi->close();
      delete fi;
    }
  if (fo != NULL)
    {
      fo->close();
      delete fo;
    }
  if (zipFi != NULL)
    {
      fclose(zipFi);
    }
  if (zipFo != NULL)
    {
      fclose(zipFo);
    }
}
int
ioFile::check(char* fileName, int out)
{
  FILE* fp;
  fp = fopen(fileName, "r");
  if (!fp)
    {
      if (out)
        {
          cerr << "Error: " << fileName << endl;
        }
      return 0;
    }
  else
    {
      fclose(fp);
      return 1;
    }
}
int
ioFile::getEOF()
{
	// for test
	//cout << "ioFile::getEOF here" << endl;
  if (compressed)
    {
	  // for test
	  //cout << "ioFile::getEOF here 1" << endl;
      return feof(zipFi);
    }
  else
    {
	  // for test
	  //cout << "ioFile::getEOF here 2 fi: " << fi << endl;
      return (*fi).eof();
    }
}

int
ioFile::takeReadFile(char* readFileName)
{
  // File exists?
  if (!check(readFileName, 1))
    {
      return 0;
    }
  // for test
  //cout << "ioFile::takeReadFile ok for file exists" << endl;
  compressed = 0;
  char *s;
  s = strstr(readFileName, ".gz");
  if (s != NULL) // compressed with gzip
    {
	  // for test
	  //cout << "ioFile::takeReadFile here" << endl;
      if (zipFi != NULL)
        {
          fclose(zipFi);
        }
      sprintf(buf, "%s %s", GUNZIP_CMD, readFileName);
      zipFi = popen(buf, "r");
      compressed = 1;
      return 1;
    }
  else
    {
	  // for test
	  //cout << "ioFile::takeReadFile here 1 fi: " << fi << endl;
	  //cout << "ioFile::takeReadFile here 1 format: " << format << endl;
      if (fi != NULL)
        {
    	  // for test
    	  //cout << "ioFile::takeReadFile here 2" << endl;
          fi->close();
          delete fi;
        }
      if (format == TEXT)
        {
    	  // for test
    	  //cout << "ioFile::takeReadFile here 3" << endl;
          fi = new ifstream(readFileName);
          // for test
          //cout << "ioFile::takeReadFile fi: " << fi << endl;
          return fi->good();
        }
      else if (format == BINARY)
        {
    	  // for test
		  //cout << "ioFile::takeReadFile here 4" << endl;
          fi = new ifstream(readFileName, ios::binary);
          // for test
          //cout << "ioFile::takeReadFile fi: " << fi << endl;
          return fi->good();
        }
    }
  return 1;
}
int
ioFile::takeWriteFile(char* writeName)
{
  compressed = 0;
  char *s;
  s = strstr(writeName, ".gz");
  if (s != NULL) // compressed with gzip
    {
      if (zipFo != NULL)
        {
          fclose(zipFo);
        }
      sprintf(buf, "%s > %s", GZIP_CMD, writeName);
      zipFo = popen(buf, "w");
      compressed = 1;
      return 1;
    }
  else
    {
      if (fo != NULL)
        {
          fo->close();
          delete fo;
        }
      if (format == TEXT)
        {
          fo = new ofstream(writeName);
        }
      //binary
      else
        {
          fo = new ofstream(writeName, ios::binary);
        }
      return fo->good();
    }
}
void
ioFile::freeWriteFile()
{
  if (fo != NULL)
    {
      fo->close();
      delete fo;
      fo = NULL;
    }
  if (zipFo != NULL)
    {
      fclose(zipFo);
      zipFo = NULL;
    }
}

void
ioFile::freeReadFile()
{
  if (fi != NULL)
    {
      fi->close();
      delete fi;
      fi = NULL;
    }
  if (zipFi != NULL)
    {
      fclose(zipFi);
      zipFi = NULL;
    }
}

#define READ(Type, type) \
int ioFile::read##Type(type& tensor##Type)\
{\
   if(compressed && (format == TEXT))\
   {\
     fscanf(zipFi, "%s", readCharBuff);\
     istringstream convertString(readCharBuff);\
     convertString >> tensor##Type;\
   }\
   else if(compressed && (format == BINARY))\
   {\
     fread((char*) &tensor##Type, sizeof(type), 1, zipFi);\
   }\
   else if (format == TEXT)\
   {\
      (*fi) >> tensor##Type;\
   }\
   else if (format == BINARY)\
   {\
      fi->read((char*) &tensor##Type, sizeof(type));\
   }\
   return 1;\
}

READ(Int, int)
READ(Long, long)
READ(Float, float)
READ(Char, char)

#define WRITE(Type, type) \
int ioFile::write##Type(type tensor##Type)\
{\
   if(compressed && (format == TEXT))\
   {\
     ostringstream convertString;\
     convertString << tensor##Type;\
     fprintf(zipFo, "%s\n", convertString.str().c_str());\
   }\
   else if(compressed && (format == BINARY))\
   {\
     fwrite((char*) &tensor##Type, sizeof(type), 1, zipFo);\
   }\
   else if (format == TEXT)\
   {\
      (*fo) << tensor##Type << "\n";\
   }\
   else\
   {\
      fo->write((char*) &tensor##Type, sizeof(type));\
   }\
   return 1;\
}

WRITE(Int, int)
WRITE(Long, long)
WRITE(Float, float)
WRITE(Char, char)

#define READ_ARRAY(Type, type) \
type* ioFile::read##Type##Array(type* tensor##Type, int n)\
{\
   if(compressed && (format == TEXT))\
   {\
      int i;\
      for (i=0; i<n; i++)\
      {\
         fscanf(zipFi, "%s", readCharBuff);\
         istringstream convertString(readCharBuff);\
         convertString >> tensor##Type[i];\
      }\
   }\
   else if(compressed && (format == BINARY))\
   {\
     fread((char*) tensor##Type, sizeof(type), n, zipFi);\
   }\
   else if(format == TEXT)\
   {\
      int i;\
      for (i=0; i<n; i++)\
      {\
        (*fi) >> tensor##Type[i];\
      }\
   }\
   else if(format == BINARY)\
   {\
      fi->read((char*) tensor##Type, n * sizeof(type));\
   }\
   return tensor##Type;\
}

READ_ARRAY(Int, int)
READ_ARRAY(Long, long)
READ_ARRAY(Float, float)
READ_ARRAY(Char, char)

#define WRITE_ARRAY(Type, type) \
int ioFile::write##Type##Array(const type* tensor##Type, int n)\
{\
   if(compressed && (format == TEXT))\
   {\
      int i;\
      for (i = 0; i < n - 1; i++)\
      {\
         ostringstream convertString;\
         convertString << tensor##Type[i];\
         fprintf(zipFo, "%s ", convertString.str().c_str());\
      }\
      ostringstream convertString;\
      convertString << tensor##Type[i];\
      fprintf(zipFo, "%s\n", convertString.str().c_str());\
   }\
   else if(compressed && (format == BINARY))\
   {\
     fwrite((char*) tensor##Type, sizeof(type), n, zipFo);\
   }\
   else if(format == TEXT)\
   {\
      int i;\
      for (i = 0; i< n - 1; i++)\
      {\
         (*fo) << tensor##Type[i] << " ";\
      }\
      (*fo) << tensor##Type[i] << "\n";\
   }\
   else\
   {\
      fo->write((char*) tensor##Type, n * sizeof(type));\
   }\
   return 1;\
}

WRITE_ARRAY(Int, int)
WRITE_ARRAY(Long, long)
WRITE_ARRAY(Float, float)
WRITE_ARRAY(Char, char)

int
ioFile::writeString(const string str)
{
  string localStr = str;
  int dvd;
  dvd = sizeof(int) - (str.size() % sizeof(int)) - 1;
  for (int i = 0; i < dvd; i++)
    {
      localStr = localStr + " ";
    }
  writeNorString(localStr);
  return 1;
}

int
ioFile::writeNorString(const string str)
{
  if (compressed && (format == TEXT))
    {
      fprintf(zipFo, "%s\n", str.c_str());
    }
  if (compressed && (format == BINARY))
    {
      fwrite(str.c_str(), 1, str.size(), zipFo);
      writeChar('\n');
    }
  else
    {
      (*fo) << str << endl;
    }
  return 1;
}

int
ioFile::readString(string& out)
{
  string str = "";
  // for test
  //cout << "ioFile::readString here" << endl;
  while (str == "" && !getEOF())
    {
	  // for test
	  //cout << "ioFile::readString here 1" << endl;
      getLine(str);
      // for test
      //cout << "ioFile::readString here 2" << endl;

    }
  // for test
  //cout << "ioFile::readString here 3" << endl;
  istringstream streamStr(str);
  // for test
  //cout << "ioFile::readString here 4" << endl;
  streamStr >> out; //Don't need blank
  return 1;
}

int
ioFile::getLine(string& out)
{
  // Only with format = 0, use with text file only,
  // If use with compressed model file, crash because the line is too long
  if (compressed)
    {
      char dataChar[MAX_CHAR_PER_SENTENCE];
      string::iterator it;

      if (fgets(dataChar, MAX_CHAR_PER_SENTENCE, zipFi) != NULL)
        {
          out = dataChar;
          it = out.end() - 1;
          out.erase(it);
          return 1;
        }
      else
        {
          out = "";
          return 0;
        }
    }
  else
    {
      if (getline(*fi, out))
        {
          return 1;
        }
      else
        {
          return 0;
        }
    }
  return 1;
}

string
ioFile::recognition(char* inputFileName)
{
  if (!check(inputFileName, 0))
    {
      return "";
    }
  else
    {
      takeReadFile(inputFileName);
      string name;
      readString(name);
      string formatStr;
      readString(format);
      freeReadFile();
      return name;
    }
}
