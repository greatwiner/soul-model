#include "text.H"

DataSet::~DataSet()
{
  if (data != NULL)
    {
      delete[] data;
    }
}

DataSet::DataSet()
{
  groupContext = 1;
  data = NULL;
  ngramNumber = 0;
  maxNgramNumber = 0;
}

void
DataSet::reset()
{
  ngramNumber = 0;
}

int
DataSet::checkBlankString(string line)
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
DataSet::resamplingDataDes(char* dataDesFileName, int type)
{
  this->type = type;
  reset();
  ioFile iofRead;
  iofRead.format = TEXT;
  int resampling = 0;
  int allLineNumber = 0;
  string line;
  int totalLineNumber = 0;
  float percent;
  char dataFileName[260];
  int resamplingLineNumber;

  //Now read
  ioFile iof;
  iof.format = TEXT;
  iofRead.takeReadFile(dataDesFileName);
  while (!iofRead.getEOF())
    {
      if (iofRead.getLine(line) && !checkBlankString(line))
        {
          istringstream ostr(line);
          ostr >> dataFileName >> totalLineNumber >> percent;
          //cout << "DataSet::resamplingDataDes line: " << line << endl;
          if (percent < 1)
            {
              resampling = 1;
            }

          resamplingLineNumber = (int) (totalLineNumber * percent);
          if (!iof.check(dataFileName, 1))
            {
              return 1;
            }
          iof.takeReadFile(dataFileName);
          cout << "read file: " << dataFileName << endl;
          resamplingText(&iof, totalLineNumber, resamplingLineNumber);
        }
    }
  return resampling;
}
