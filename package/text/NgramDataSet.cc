/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for data set of n-gram models.
 See DataSet.H & NgramDataSet.H for more detail
 *******************************************************************/

#include "text.H"
NgramDataSet::NgramDataSet(int type, int n, int BOS, SoulVocab* inputVoc,
    SoulVocab* outputVoc, int mapIUnk, int mapOUnk, int maxNgramNumber)
{
  this->type = type;
  this->n = n;
  this->BOS = BOS;
  // BOS cannot be greater than n - 1
  if (this->BOS > n - 1)
    {
      this->BOS = this->n - 1;
    }
  this->inputVoc = inputVoc;
  this->outputVoc = outputVoc;
  this->mapIUnk = mapIUnk;
  this->mapOUnk = mapOUnk;
  data = NULL;
  ngramNumber = 0;
  this->maxNgramNumber = maxNgramNumber;
  // Treat to have memory for size, stride
  dataTensor.resize(1, 1);

  // Ask for memory for data
  // n + 3 because we have ID_END_NGRAM and two ints to encode info
  try
    {
      data = new int[maxNgramNumber * (n + 3)];
    }
  catch (bad_alloc& ba)
    {
      cerr << "NgramDataSet bad_alloc caught: " << ba.what() << endl;
      exit(1);
    }
}

NgramDataSet::NgramDataSet(int n, int maxNgramNumber)
{

  this->n = n;

  data = NULL;
  ngramNumber = 0;
  this->maxNgramNumber = maxNgramNumber;
  dataTensor.resize(1, 1);
  try
    {
      data = new int[maxNgramNumber * (n + 3)];
    }
  catch (bad_alloc& ba)
    {
      cerr << "bad_alloc caught: " << ba.what() << endl;
    }
}

int
NgramDataSet::addLine(string line)
{
  int j;
  int inputIndex[MAX_WORD_PER_SENTENCE];
  int outputIndex[MAX_WORD_PER_SENTENCE];
  istringstream streamLine(line);
  string word;
  int i = 0;
  int length = 0;
  int use;
  while (streamLine >> word)
    {
      // Have word index for input and output vocabs
      inputIndex[length] = inputVoc->index(word);
      outputIndex[length] = outputVoc->index(word);
      // If map?Unk = 1, considering unknown words as UNK
      if (mapIUnk && inputIndex[length] == ID_UNK)
        {
          inputIndex[length] = inputVoc->unk;
        }
      if (mapOUnk && outputIndex[length] == ID_UNK)
        {
          outputIndex[length] = outputVoc->unk;
        }
      length++;
      // Line is too long, cut
      if (length >= MAX_WORD_PER_SENTENCE)
        {
          cerr << "WARNING: Line " << line << " is too long" << endl;
          return 0;
        }
    }

  // Line has no ngram, don't do anything
  // length = BOS - 1 because (n - 1) <s> has been added before
  if (length == BOS - 1)
    {
      cerr << "WARNING: Line " << line << " is too short" << endl;
      return 0;
    }
  // Check if n-grams in the sentence satisfy
  for (i = 0; i <= length - n; i++)
    {
      use = 1;
      // Cases for left to right (0) or right to left (1)
      if (type == 0 || type == 1)
        {
          // Unknown words or not map to UNK
          for (j = 0; j < n - 1; j++)
            {
              if (inputIndex[i + j] < 0)
                {
                  use = 0;
                  break;
                }
              data[ngramNumber * (n + 3) + j] = inputIndex[i + j];
            }
          if (outputIndex[i + n - 1] < 0)
            {
              use = 0;
            }
          else
            {
              data[ngramNumber * (n + 3) + n - 1] = outputIndex[i + n - 1];
            }
        }
      // Case for models that predict the center word
      // at (n - 1 / 2) position
      else
        {
          for (j = 0; j < n; j++)
            {
              // Unknown words or not map to UNK
              if (inputIndex[i + j] < 0 && j != (n - 1) / 2)
                {
                  use = 0;
                  break;
                }
              if (j < (n - 1) / 2)
                {
                  data[ngramNumber * (n + 3) + j] = inputIndex[i + j];
                }
              else if (j > (n - 1) / 2)
                {
                  data[ngramNumber * (n + 3) + j - 1] = inputIndex[i + j];
                }
            }
          if (outputIndex[i + (n - 1) / 2] < 0)
            {
              use = 0;
            }
          else
            {
              data[ngramNumber * (n + 3) + n - 1]
                  = outputIndex[i + (n - 1) / 2];
            }
        }
      if (use)
        {
          data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
          // Normally, keep the order of this n-gram in file
          data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
          // The value to find the next different context for context grouping
          data[ngramNumber * (n + 3) + n + 2] = 0;
          ngramNumber++;
        }
    }
  return 1;
}

int
NgramDataSet::resamplingSentence(int totalLineNumber, int resamplingLineNumber,
    int* resamplingLineId)
{
  // Don't resample, use all possible sentences
  if (totalLineNumber == resamplingLineNumber)
    {
      int i;
      for (i = 0; i < totalLineNumber; i++)
        {
          resamplingLineId[i] = i;
        }
      return 1;
    }
  // Resampling indices of sentences using rand(), maybe not good enough
  else
    {
      int* buff = new int[totalLineNumber];
      int chosenPos;
      int i;
      for (i = 0; i < totalLineNumber; i++)
        {
          buff[i] = i;
        }
      int pos = totalLineNumber;
      for (i = 0; i < resamplingLineNumber; i++)
        {
          chosenPos = rand() % pos;
          resamplingLineId[i] = buff[chosenPos];
          buff[chosenPos] = buff[pos - 1];
          pos--;
        }
      delete[] buff;
      sort(resamplingLineId, resamplingLineId + resamplingLineNumber);
      return 1;
    }
}

int
NgramDataSet::readText(ioFile* iof)
{
  int i = 0;
  string line;
  string headline;
  string invLine;
  string tailline;
  headline = "";
  tailline = "";
  // Add <s>, </s> depending on model type
  // Normal
  if (type == 0)
    {
      for (i = 0; i < BOS; i++)
        {
          headline = headline + SS + " ";
        }
      tailline = tailline + " " + ES;
    }
  // Inverse
  else if (type == 1)
    {
      for (i = 0; i < BOS; i++)
        {
          tailline = tailline + " " + ES;
        }
      headline = headline + SS + " ";

    }
  // Center
  else if (type == 2)
    {
      for (i = 0; i < BOS / 2; i++)
        {
          tailline = tailline + " " + ES;
          headline = headline + SS + " ";
        }
    }
  int readLineNumber = 0;
  while (!iof->getEOF())
    {
      if (iof->getLine(line))
        {
          // Don't use empty line
          if (!checkBlankString(line))
            {
              line = headline + line + tailline;
              if (type == 0 || type == 2)
                {
                  addLine(line);
                }
              else if (type == 1)
                {
                  invLine = inverse(line);
                  addLine(invLine);
                }
            }
          else
            {
              cerr << "WARNING: Line " << line << " is empty" << endl;
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
  return 1;
}

int
NgramDataSet::resamplingText(ioFile* iof, int totalLineNumber,
    int resamplingLineNumber)
{
  int* resamplingLineId = new int[resamplingLineNumber];
  resamplingSentence(totalLineNumber, resamplingLineNumber, resamplingLineId);

  int i = 0;
  string line;
  string headline;
  string invLine;
  string tailline;
  headline = "";
  tailline = "";
  // Normal
  if (type == 0)
    {
      for (i = 0; i < BOS; i++)
        {
          headline = headline + SS + " ";
        }
      tailline = tailline + " " + ES;
    }
  // Inverse
  else if (type == 1)
    {
      for (i = 0; i < BOS; i++)
        {
          tailline = tailline + " " + ES;
        }
      headline = headline + SS + " ";

    }
  // Center
  else if (type == 2)
    {
      for (i = 0; i < BOS / 2; i++)
        {
          tailline = tailline + " " + ES;
          headline = headline + SS + " ";
        }
    }
  int readLineNumber = 0;
  int currentId = 0;
  while (!iof->getEOF())
    {
      if (iof->getLine(line))
        {
          // Use only lines with index in the resampling list
          if (readLineNumber == resamplingLineId[currentId])
            {
              if (!checkBlankString(line))
                {
                  line = headline + line + tailline;
                  if (type == 0 || type == 2)
                    {
                      addLine(line);
                    }
                  else if (type == 1)
                    {
                      invLine = inverse(line);
                      addLine(invLine);
                    }
                }
              currentId++;
            }
          if (currentId == resamplingLineNumber)
            {
              break;
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
  delete[] resamplingLineId;
  return ngramNumber;

}

int
NgramDataSet::readTextNgram(ioFile* iof)
{
  string line;
  string invLine;
  int readLineNumber = 0;
  while (!iof->getEOF())
    {
      if (iof->getLine(line))
        {
          if (!checkBlankString(line))
            {
              if (type == 0 || type == 2)
                {
                  addLine(line);
                }
              else if (type == 1)
                {
                  invLine = inverse(line);
                  addLine(invLine);
                }
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
  return ngramNumber;
}

int
NgramDataSet::readCoBiNgram(ioFile* iof)
{
  int readLineNumber = 0;
  int i;
  int N;
  iof->readInt(N);
  int readTextNgram[N];
  // Order in the file can be larger than order of model
  int offset = N - n;
  if (offset < 0)
    {
      cerr << "ERROR: order in id file is too small:" << N << " < " << n
          << endl;
      exit(1);
    }
  while (!iof->getEOF())
    {
      iof->readIntArray(readTextNgram, N);
      if (iof->getEOF())
        {
          break;
        }
      for (i = 0; i < n; i++)
        {
          data[ngramNumber * (n + 3) + i] = readTextNgram[offset + i];
        }
      data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
      data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
      data[ngramNumber * (n + 3) + n + 2] = 0;
      ngramNumber++;
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
  return ngramNumber;
}

intTensor&
NgramDataSet::createTensor()
{
  dataTensor.haveMemory = 0;
  dataTensor.size[0] = ngramNumber;
  dataTensor.size[1] = n + 3;
  dataTensor.stride[0] = n + 3;
  dataTensor.stride[1] = 1;
  // dataTensor is a pointer, doesn't have data
  if (dataTensor.data != data)
    {
      delete[] dataTensor.data;
      dataTensor.data = data;
    }
  // Sort using quicksort
  if (groupContext)
    {
      sortNgram();
    }
  // Edit info integer to keep the info for the first next n-gram
  // which has a different context
  int ngramId;
  int preNgramId = 0;
  int i;
  int equal = 1;
  for (ngramId = 0; ngramId < ngramNumber - 1; ngramId++)
    {
      equal = 1;
      for (i = 0; i < n - 1; i++)
        {

          if (data[ngramId * (n + 3) + i] != data[(ngramId + 1) * (n + 3) + i])
            {
              equal = 0;
              break;
            }
        }
      if (equal == 0 || !groupContext)
        {
          data[preNgramId * (n + 3) + n + 2] = ngramId + 1;
          preNgramId = ngramId + 1;
        }
    }
  if (equal == 1)
    {
      data[preNgramId * (n + 3) + n + 2] = ngramNumber;
    }
  data[ngramNumber * (n + 3) - 1] = ngramNumber;
  probTensor.resize(ngramNumber, 1);
  return dataTensor;
}

void
NgramDataSet::writeReBiNgram(ioFile* iof)
{
  iof->writeInt(ngramNumber);
  iof->writeInt(n);
  int ngramId = 0;
  for (ngramId = 0; ngramId < ngramNumber; ngramId++)
    {
      iof->writeIntArray(data + ngramId * (n + 3), n);
    }
}

string
NgramDataSet::inverse(string line)
{
  istringstream streamLine(line);
  string word;
  string newLine = "";
  while (streamLine >> word)
    {
      newLine = word + " " + newLine;
    }
  return newLine;
}

int
compare(const void *ngram1, const void *ngram2)
{

  int i;
  int *pNgram1;
  int *pNgram2;

  pNgram1 = (int *) ngram1;
  pNgram2 = (int *) ngram2;
  i = 0;
  do
    {

      if (pNgram1[i] < pNgram2[i])
        {
          return -1;
        }
      else
        {
          if (pNgram1[i] > pNgram2[i])
            {
              return 1;
            }
        }
      i++;
    }
  while (pNgram1[i] != ID_END_NGRAM);
  return 0;

}

void
NgramDataSet::sortNgram()
{
  // Use quicksort to actually compare n parameters,
  // the index is kept in the n + 2 element
  qsort((void*) data, (size_t) ngramNumber, (n + 3) * sizeof(unsigned int),
      compare);
}
void
NgramDataSet::shuffle(int times)
{
  int n3 = n + 3;
  int *tg = new int[n3 * sizeof(int)];
  int i;
  int p1, p2;
  for (i = 0; i < times * ngramNumber; i++)
    {
      p1 = (int) (ngramNumber * drand48());
      p2 = (int) (ngramNumber * drand48());
      memcpy(tg, data + p1 * n3, n3 * sizeof(int));
      memcpy(data + p1 * n3, data + p2 * n3, n3 * sizeof(int));
      memcpy(data + p2 * n3, tg, n3 * sizeof(int));
    }
}

float
NgramDataSet::computePerplexity()
{
  perplexity = 0;
  for (int i = 0; i < probTensor.length; i++)
    {
      perplexity += log(probTensor(i));
    }
  perplexity = exp(-perplexity / ngramNumber);
  return perplexity;
}

int
NgramDataSet::addLine(ioFile* iof)
{
  string line;
  iof->getLine(line);
  addLine(line);
  return 1;
}

int
NgramDataSet::resamplingIdDataDes(char* dataDesFileName)
{
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
  iof.format = BINARY;
  iofRead.takeReadFile(dataDesFileName);
  while (!iofRead.getEOF())
    {
      if (iofRead.getLine(line) && line != "")
        {
          istringstream ostr(line);
          ostr >> dataFileName >> totalLineNumber >> percent;
          resamplingLineNumber = (int) (totalLineNumber * percent);
          if (percent < 1)
            {
              resampling = 1;
            }
          iof.takeReadFile(dataFileName);
          cout << "read file: " << dataFileName << endl;
          resamplingId(&iof, totalLineNumber, resamplingLineNumber);
        }
    }
  return resampling;
}

int
NgramDataSet::resamplingId(ioFile* iof, int totalLineNumber,
    int resamplingLineNumber)
{
  int* resamplingLineId = new int[resamplingLineNumber];
  resamplingSentence(totalLineNumber, resamplingLineNumber, resamplingLineId);

  int i = 0;
  int readLineNumber = 0;
  int currentId = 0;

  int N;
  iof->readInt(N);
  int readTextNgram[N];
  int offset = N - n;
  if (offset < 0)
    {
      cerr << "ERROR: order in id file is too small:" << N << " < " << n
          << endl;
      exit(1);
    }
  while (!iof->getEOF())
    {
      iof->readIntArray(readTextNgram, N);
      if (iof->getEOF())
        {
          break;
        }
      if (readLineNumber == resamplingLineId[currentId])
        {

          for (i = 0; i < n; i++)
            {
              data[ngramNumber * (n + 3) + i] = readTextNgram[offset + i];
            }
          data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
          data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
          data[ngramNumber * (n + 3) + n + 2] = 0;
          ngramNumber++;

          currentId++;
          if (currentId == resamplingLineNumber)
            {
              break;
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
  delete[] resamplingLineId;
  return ngramNumber;
}
