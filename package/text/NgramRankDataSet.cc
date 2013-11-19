/*******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram ranking data set.
 See DataSet.H and NgramRankDataSet.H for more detail
 *******************************************************************/
#include "text.H"
NgramRankDataSet::NgramRankDataSet(int type, int n, int BOS,
    SoulVocab* inputVoc, SoulVocab* outputVoc, int mapIUnk, int mapOUnk,
    int maxNgramNumber)
{
  srand48(time(NULL));
  srand(time(NULL));
  this->type = type;
  this->n = n;
  this->BOS = BOS;
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
  dataTensor.resize(1, 1);
  try
    {
      data = new int[maxNgramNumber * (n + 3)];
    }
  catch (bad_alloc& ba)
    {
      cerr << "bad_alloc caught: " << ba.what() << endl;
      exit(1);
    }
}

int
NgramRankDataSet::addLine(string line)
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
      inputIndex[length] = inputVoc->index(word);
      outputIndex[length] = outputVoc->index(word);
      if (mapIUnk && inputIndex[length] == ID_UNK)
        {
          inputIndex[length] = inputVoc->unk;
        }
      if (mapOUnk && outputIndex[length] == ID_UNK)
        {
          outputIndex[length] = outputVoc->unk;
        }
      length++;
      // Line is too long
      if (length > MAX_WORD_PER_SENTENCE)
        {
          return 0;
        }
    }
  // Line has no ngram, don't do anything
  if (length == BOS - 1)
    {
      return 0;
    }
  for (i = 0; i <= length - n; i++)
    {
      use = 1;
      if (type == 0 || type == 1)
        {

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
      else
        {
          for (j = 0; j < n; j++)
            {
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
          data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
          data[ngramNumber * (n + 3) + n + 2] = 0;
          // for test
          /*cout << "data positive: " << endl;
          for (j = 0; j < n+3; j ++) {
        	  cout << data[ngramNumber * (n+3) + j] << " ";
          }
          cout << endl;*/
          ngramNumber++;
          //Add negative example, copy context and random uniform predicted word
          for (j = 0; j < n - 1; j++)
            {
              data[ngramNumber * (n + 3) + j] = data[(ngramNumber - 1)
                  * (n + 3) + j];
            }
          do
            {
              data[ngramNumber * (n + 3) + n - 1]
                  = (int) (outputVoc->wordNumber * drand48());
            }
          while (data[ngramNumber * (n + 3) + n - 1] == data[(ngramNumber - 1)
              * (n + 3) + n - 1]);
          data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
          data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
          data[ngramNumber * (n + 3) + n + 2] = 0;
          // for test
          /*cout << "data negative: " << endl;
          for (j = 0; j < n+3; j ++) {
        	  cout << data[ngramNumber * (n+3) + j] << " ";
          }
          cout << endl;*/
          ngramNumber++;
        }
    }
  return 1;
}

int
NgramRankDataSet::resamplingSentence(int totalLineNumber,
    int resamplingLineNumber, int* resamplingLineId)
{
  if (totalLineNumber == resamplingLineNumber)
    {
      int i;
      for (i = 0; i < totalLineNumber; i++)
        {
          resamplingLineId[i] = i;
        }
      return 1;
    }
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
NgramRankDataSet::readText(ioFile* iof)
{
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
  while (!iof->getEOF())
    {
      if (iof->getLine(line))
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
NgramRankDataSet::resamplingText(ioFile* iof, int totalLineNumber,
    int resamplingLineNumber)
{
  int* resamplingLineId = new int[resamplingLineNumber];
  resamplingSentence(totalLineNumber, resamplingLineNumber, resamplingLineId);
  // for test

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
    	  // for test
    	  //cout << "NgramRankDataSet::resamplingText line: " << line << endl;
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

intTensor&
NgramRankDataSet::createTensor()
{
  dataTensor.haveMemory = 0;
  dataTensor.size[0] = ngramNumber;
  dataTensor.size[1] = n + 3;
  dataTensor.stride[0] = n + 3;
  dataTensor.stride[1] = 1;
  if (dataTensor.data != data)
    {
      delete[] dataTensor.data;
      dataTensor.data = data;
    }
  int ngramId;
  int preNgramId = 0;
  for (ngramId = 0; ngramId < ngramNumber - 1; ngramId++)
    {
      data[preNgramId * (n + 3) + n + 2] = ngramId + 1;
      preNgramId = ngramId + 1;
    }
  data[ngramNumber * (n + 3) - 1] = ngramNumber;
  probTensor.resize(ngramNumber, 1);
  return dataTensor;
}

int
NgramRankDataSet::readTextNgram(ioFile* iof)
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
NgramRankDataSet::readCoBiNgram(ioFile* iof)
{
  int readLineNumber = 0;
  int i;
  int N;
  int ngramNumberInFile;
  iof->readInt(ngramNumberInFile);
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
      for (i = 0; i < n; i++)
        {
          data[ngramNumber * (n + 3) + i] = readTextNgram[offset + i];
        }
      data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
      data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
      data[ngramNumber * (n + 3) + n + 2] = 0;
      // for test
      //cout << "NgramRankDataSet::readCoBiNgram ngramNumber: " << ngramNumber << endl;
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

void
NgramRankDataSet::writeReBiNgram(ioFile* iof)
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
NgramRankDataSet::inverse(string line)
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
rankCompare(const void *ngram1, const void *ngram2)
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
NgramRankDataSet::sortNgram()
{
  qsort((void*) data, (size_t) ngramNumber, (n + 3) * sizeof(unsigned int),
      rankCompare);

}
void
NgramRankDataSet::shuffle(int times)
{
  int n3 = n + 3;
  int *tg = new int[2 * n3 * sizeof(int)];
  int i;
  int p1, p2;
  for (i = 0; i < times * ngramNumber; i++)
    {
      do
        {
          p1 = (int) (ngramNumber * drand48());
        }
      while (p1 % 2 != 0);
      do
        {
          p2 = (int) (ngramNumber * drand48());
        }
      while (p2 % 2 != 0);
      memcpy(tg, data + p1 * n3, 2 * n3 * sizeof(int));
      memcpy(data + p1 * n3, data + p2 * n3, 2 * n3 * sizeof(int));
      memcpy(data + p2 * n3, tg, 2 * n3 * sizeof(int));
    }
}

int
NgramRankDataSet::writeReBiNgram()
{

  int i;
  int ngramId = 0;
  for (ngramId = 0; ngramId < ngramNumber; ngramId++)
    {
      for (i = 0; i < n + 3; i++)
        {
          cout << data[ngramId * (n + 3) + i] << " ";
        }
      cout << endl;
    }
  return 1;
}

float
NgramRankDataSet::computePerplexity()
{
  perplexity = 0;
  for (int i = 0; i < probTensor.length; i++)
    {
      perplexity += probTensor(i);
      // for test
      //cout << "NgramRankDataSet::computePerplexity perplexity: " << perplexity << endl;
    }
  perplexity = perplexity / ngramNumber * 2;
  return perplexity;
}

int
NgramRankDataSet::addLine(ioFile* iof)
{
  string line;
  iof->getLine(line);
  addLine(line);
  return 1;
}

