/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram phrase based data set.
 *******************************************************************/
#include "text.H"
NgramPhraseTranslationDataSet::NgramPhraseTranslationDataSet(int type, int n,
    int BOS, SoulVocab* inputVoc, SoulVocab* outputVoc, int mapIUnk,
    int mapOUnk, int maxNgramNumber)
{
  this->type = type;
  this->n = n;
  this->BOS = BOS;
  this->inputVoc = inputVoc;
  this->outputVoc = outputVoc;
  this->mapIUnk = mapIUnk;
  this->mapOUnk = mapOUnk;
  data = NULL;
  ngramNumber = 0;
  this->maxNgramNumber = maxNgramNumber;
  dataTensor.resize(1, 1);

  if (type == 0 || type == 2)
    {
      nm = n * 2;
    }
  else
    {
      nm = n * 2 - 1;
    }

  try
    {
      data = new int[maxNgramNumber * (nm + 3)];
    }
  catch (bad_alloc& ba)
    {
      cerr << "bad_alloc caught: " << ba.what() << endl;
      exit(1);
    }
  if (this->BOS > n - 1)
    {
      this->BOS = n - 1;
    }
}

int
NgramPhraseTranslationDataSet::resamplingSentence(int totalLineNumber,
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
NgramPhraseTranslationDataSet::addLine(ioFile* iof)
{
  int i = 0;
  string line;
  string headline;
  headline = "";
  int inputIndex[MAX_WORD_PER_SENTENCE];
  int outputIndex[MAX_WORD_PER_SENTENCE];
  //unkIndex for predicted stuff
  int unkIndex[MAX_WORD_PER_SENTENCE];
  int inputLength = 0;
  int outputLength = 0;
  string word;
  int count;
  string tuple;
  string preSrc = PREFIX_SOURCE;

  if (iof->getLine(line))
    {
      for (i = 0; i < BOS; i++)
        {
          inputIndex[inputLength] = inputVoc->ss;
          outputIndex[outputLength] = inputVoc->ss;
          unkIndex[outputLength] = inputVoc->ss;
          inputLength++;
          outputLength++;
        }

      do
        {
          istringstream streamLine(line);
          streamLine >> word;

          count = 0;
          while (word != "|||")
            {
              if (count == 0)
                {
                  tuple = preSrc + word;
                }
              else
                {
                  tuple = tuple + " " + word;
                }
              count = count + 1;
              streamLine >> word;
            }
          inputIndex[inputLength] = inputVoc->index(tuple);
          if (mapIUnk && inputIndex[inputLength] == ID_UNK)
            {
              inputIndex[inputLength] = inputVoc->unk;
            }
          if (type == 2 || type == 3)
            {
              unkIndex[inputLength] = outputVoc->index(tuple);
              if (mapOUnk && (unkIndex[inputLength] == ID_UNK))
                {
                  unkIndex[inputLength] = outputVoc->unk;
                }
            }

          inputLength++;

          count = 0;
          while (streamLine >> word)
            {
              if (count == 0)
                {
                  tuple = word;
                }
              else
                {
                  tuple = tuple + " " + word;
                }
              count = count + 1;
            }
          outputIndex[outputLength] = inputVoc->index(tuple);
          if (mapIUnk && outputIndex[outputLength] == ID_UNK)
            {
              outputIndex[outputLength] = inputVoc->unk;
            }
          if (type == 0 || type == 1)
            {
              unkIndex[outputLength] = outputVoc->index(tuple);
              if (mapOUnk && (unkIndex[outputLength] == ID_UNK))
                {
                  unkIndex[outputLength] = outputVoc->unk;
                }
            }
          outputLength++;
          addDisTuple(inputIndex + inputLength - n,
              outputIndex + outputLength - n, unkIndex + outputLength - n);
          iof->getLine(line);
        }
      while (line != "EOS");
      inputIndex[inputLength] = inputVoc->es;
      outputIndex[outputLength] = inputVoc->es;
      unkIndex[outputLength] = outputVoc->es;
      inputLength++;
      outputLength++;
      addDisTuple(inputIndex + inputLength - n, outputIndex + outputLength - n,
          unkIndex + outputLength - n);
    }
  return ngramNumber;
}
int
NgramPhraseTranslationDataSet::readText(ioFile* iof)
{
  int readLineNumber = 0;
  while (!iof->getEOF())
    {
      addLine(iof);
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
NgramPhraseTranslationDataSet::resamplingText(ioFile* iof, int totalLineNumber,
    int resamplingLineNumber)
{
  int* resamplingLineId = new int[resamplingLineNumber];
  resamplingSentence(totalLineNumber, resamplingLineNumber, resamplingLineId);

  int readLineNumber = 0;
  int currentId = 0;
  string line;
  while (!iof->getEOF())
    {
      if (readLineNumber != resamplingLineId[currentId])
        {
          line = "";
          while (line != "EOS")
            {
              iof->getLine(line);
            }
        }
      else
        {
          currentId++;
          addLine(iof);
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
NgramPhraseTranslationDataSet::createTensor()
{
  dataTensor.haveMemory = 0;
  dataTensor.size[0] = ngramNumber;
  dataTensor.size[1] = nm + 3;
  dataTensor.stride[0] = nm + 3;
  dataTensor.stride[1] = 1;
  if (dataTensor.data != data)
    {
      delete[] dataTensor.data;
      dataTensor.data = data;
    }
  if (groupContext)
    {
      sortNgram();
    }

  int ngramId;
  int preNgramId = 0;
  int i;
  int equal = 1;
  for (ngramId = 0; ngramId < ngramNumber - 1; ngramId++)
    {
      equal = 1;
      for (i = 0; i < nm - 1; i++)
        {

          if (data[ngramId * (nm + 3) + i]
              != data[(ngramId + 1) * (nm + 3) + i])
            {
              equal = 0;
              break;
            }
        }
      if (equal == 0 || !groupContext)
        {
          data[preNgramId * (nm + 3) + nm + 2] = ngramId + 1;
          preNgramId = ngramId + 1;
        }
    }
  if (equal == 1)
    {
      data[preNgramId * (nm + 3) + nm + 2] = ngramNumber;
    }
  data[ngramNumber * (nm + 3) - 1] = ngramNumber;
  probTensor.resize(ngramNumber, 1);
  return dataTensor;
}

int
NgramPhraseTranslationDataSet::readTextNgram(ioFile* iof)
{
  cerr << "Wrong call" << endl;
  return 1;
}

int
NgramPhraseTranslationDataSet::readCoBiNgram(ioFile* iof)
{
  int readLineNumber = 0;
  int i;
  int N;
  iof->readInt(N);
  int readTextNgram[N];
  int offset = N - nm;
  if (offset < 0)
    {
      cerr << "ERROR: order in id file is too small:" << N << " < " << nm
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
      for (i = 0; i < nm; i++)
        {
          data[ngramNumber * (nm + 3) + i] = readTextNgram[offset + i];
        }
      data[ngramNumber * (nm + 3) + nm] = ID_END_NGRAM;
      data[ngramNumber * (nm + 3) + nm + 1] = ngramNumber;
      data[ngramNumber * (nm + 3) + nm + 2] = 0;
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
NgramPhraseTranslationDataSet::writeReBiNgram(ioFile* iof)
{
  iof->writeInt(ngramNumber);
  iof->writeInt(nm);
  int ngramId = 0;
  for (ngramId = 0; ngramId < ngramNumber; ngramId++)
    {
      iof->writeIntArray(data + ngramId * (nm + 3), nm);
    }
}

int
NgramPhraseTranslationDataSet::addDisTuple(int* srcIndex, int* desIndex,
    int* unkIndex)
{
  int i;
  int use = 1;
  if (unkIndex[n - 1] == ID_UNK)
    {
      use = 0;
    }
  else
    {
      if (type == 0)
        {
          for (i = 0; i < n; i++)
            {
              if (srcIndex[i] == ID_UNK || desIndex[i] == ID_UNK)
                {
                  use = 0;
                  break;
                }
              data[ngramNumber * (nm + 3) + i] = srcIndex[i];
              data[ngramNumber * (nm + 3) + n + i] = desIndex[i];
            }
        }
      else if (type == 2)
        {
          for (i = 0; i < n; i++)
            {
              if (srcIndex[i] == ID_UNK || desIndex[i] == ID_UNK)
                {
                  use = 0;
                  break;
                }
              data[ngramNumber * (nm + 3) + i] = desIndex[i];
              data[ngramNumber * (nm + 3) + n + i] = srcIndex[i];
            }
        }
      else if (type == 1)
        {

          for (i = 0; i < n; i++)
            {
              if (srcIndex[i] == ID_UNK || desIndex[i] == ID_UNK)
                {
                  use = 0;
                  break;
                }
              if (i != n - 1)
                {
                  data[ngramNumber * (nm + 3) + i] = srcIndex[i];
                }
              data[ngramNumber * (nm + 3) + n - 1 + i] = desIndex[i];
            }
        }
      else if (type == 3)
        {
          for (i = 0; i < n; i++)
            {
              if (srcIndex[i] == ID_UNK || desIndex[i] == ID_UNK)
                {
                  use = 0;
                  break;
                }
              if (i != n - 1)
                {
                  data[ngramNumber * (nm + 3) + i] = desIndex[i];
                }
              data[ngramNumber * (nm + 3) + n - 1 + i] = srcIndex[i];
            }
        }
      if (use)
        {
          data[ngramNumber * (nm + 3) + nm - 1] = unkIndex[n - 1];
          data[ngramNumber * (nm + 3) + nm] = ID_END_NGRAM;
          //Normaly, is the order of this ngram in file
          data[ngramNumber * (nm + 3) + nm + 1] = ngramNumber;
          //The value to find the indentical context and ...
          data[ngramNumber * (nm + 3) + nm + 2] = 0;
          ngramNumber++;
        }
    }
  return ngramNumber;
}

int
disTupleCompare(const void *ngram1, const void *ngram2)
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
NgramPhraseTranslationDataSet::sortNgram()
{
  qsort((void*) data, (size_t) ngramNumber, (nm + 3) * sizeof(unsigned int),
      disTupleCompare);
}
void
NgramPhraseTranslationDataSet::shuffle(int times)
{
  int n3 = nm + 3;
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

void
NgramPhraseTranslationDataSet::writeReBiNgram()
{

  int i;
  int ngramId = 0;
  for (ngramId = 0; ngramId < ngramNumber; ngramId++)
    {
      for (i = 0; i < nm + 3; i++)
        {
          cout << data[ngramId * (nm + 3) + i] << " ";
        }
      cout << endl;
    }
}

float
NgramPhraseTranslationDataSet::computePerplexity()
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
NgramPhraseTranslationDataSet::addLine(string line)
{
  return 1;
}

