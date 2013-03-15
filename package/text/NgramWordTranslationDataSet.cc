/*******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram word based data set.
 *******************************************************************/
#include "text.H"
NgramWordTranslationDataSet::NgramWordTranslationDataSet(int type, int n,
    int BOS, SoulVocab* inputVoc, SoulVocab* outputVoc, int mapIUnk,
    int mapOUnk, int maxNgramNumber)
{
  this->type = type;
  this->n = n;
  nm = n * 2;
  this->BOS = BOS;
  if (this->BOS > n)
    {
      this->BOS = this->n;
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
      data = new int[maxNgramNumber * (nm + 3)];
    }
  catch (bad_alloc& ba)
    {
      cerr << "bad_alloc caught: " << ba.what() << endl;
      exit(1);
    }
}

int
NgramWordTranslationDataSet::resamplingSentence(int totalLineNumber,
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
NgramWordTranslationDataSet::addLine(ioFile* iof)
{
  int i = 0;
  string line;
  string headline;
  headline = "";
  int currentId = 0;
  int inputIndex[MAX_WORD_PER_SENTENCE];
  int outputIndex[MAX_WORD_PER_SENTENCE];
  int unkIndex[MAX_WORD_PER_SENTENCE];
  int inputLength = 0;
  int outputLength = 0;
  string word;
  int inputCount;
  int outputCount;
  // PREFIX_SOURCE is normally 'src.' to distinguish between source words
  // and target words, for example comma: , and src.,
  string preSrc = PREFIX_SOURCE;
  if (iof->getLine(line))
    {
      // Is this test really necessary :S
      if (line == "EOS")
        {
          return ngramNumber;
        }
      currentId++;
      // Initialize with SS token
      for (i = 0; i < BOS; i++)
        {
          inputIndex[inputLength] = inputVoc->ss;
          outputIndex[outputLength] = inputVoc->ss;
          unkIndex[outputLength] = outputVoc->ss;
          inputLength++;
          outputLength++;
        }

      do
        {
          istringstream streamLine(line);
          streamLine >> word;
          inputCount = 0;
          // Read source words until meet separator |||
          while (word != "|||")
            {
              inputIndex[inputLength] = inputVoc->index(preSrc + word);
              if (mapIUnk && inputIndex[inputLength] == ID_UNK)
                {
                  inputIndex[inputLength] = inputVoc->unk;
                }
              // For SrcTrg, Src models, they are also predicted words
              if (type == 2 || type == 3)
                {
                  unkIndex[inputLength] = outputVoc->index(preSrc + word);
                  if (mapOUnk && (unkIndex[inputLength] == ID_UNK))
                    {
                      unkIndex[inputLength] = outputVoc->unk;
                    }
                }
              inputLength++;
              inputCount++;
              streamLine >> word;
            }

          outputCount = 0;
          // Read target words
          while (streamLine >> word)
            {
              outputIndex[outputLength] = inputVoc->index(word);
              if (mapIUnk && (outputIndex[outputLength] == ID_UNK))
                {
                  outputIndex[outputLength] = inputVoc->unk;
                }
              // For Trg, TrgSrc models, they are also predicted words
              if (type == 0 || type == 1)
                {
                  unkIndex[outputLength] = outputVoc->index(word);
                  if (mapOUnk && (unkIndex[outputLength] == ID_UNK))
                    {
                      unkIndex[outputLength] = outputVoc->unk;
                    }
                }
              outputLength++;
              outputCount++;
            }
          // Depending on the type, add n-gram with appropriate position
          if (type == 0)
            {
              for (i = 0; i < outputCount; i++)
                {
                  addDisWordTuple(inputIndex + inputLength - n,
                      outputIndex + outputLength - outputCount + i + 1 - n,
                      unkIndex + outputLength - outputCount + i + 1 - n);
                }
            }
          else if (type == 2)
            {
              for (i = 0; i < inputCount; i++)
                {
                  addDisWordTuple(outputIndex + outputLength - n,
                      inputIndex + inputLength - inputCount + i + 1 - n,
                      unkIndex + inputLength - inputCount + i + 1 - n);
                }
            }
          else if (type == 1)
            {
              for (i = 0; i < outputCount; i++)
                {
                  addDisWordTuple(inputIndex + inputLength - n - inputCount,
                      outputIndex + outputLength - outputCount + i + 1 - n,
                      unkIndex + outputLength - outputCount + i + 1 - n);
                }
            }
          else if (type == 3)
            {
              for (i = 0; i < inputCount; i++)
                {
                  addDisWordTuple(outputIndex + outputLength - n - outputCount,
                      inputIndex + inputLength - inputCount + i + 1 - n,
                      unkIndex + inputLength - inputCount + i + 1 - n);
                }
            }
          iof->getLine(line);
        }
      while (line != "EOS");
      // Finish one sentence with ES, adding n-gram ending with ES
      inputIndex[inputLength] = inputVoc->es;
      outputIndex[outputLength] = inputVoc->es;
      if (type == 0 || type == 1)
        {
          unkIndex[outputLength] = outputVoc->es;
          if (mapOUnk && (unkIndex[outputLength] == ID_UNK))
            {
              unkIndex[outputLength] = outputVoc->unk;
            }
        }
      else if (type == 2 || type == 3)
        {
          unkIndex[inputLength] = outputVoc->es;
          if (mapOUnk && (unkIndex[inputLength] == ID_UNK))
            {
              unkIndex[inputLength] = outputVoc->unk;
            }
        }
      inputLength++;
      outputLength++;
      if (type == 0)
        {
          addDisWordTuple(inputIndex + inputLength - n,
              outputIndex + outputLength - n, unkIndex + outputLength - n);
        }
      else if (type == 2)
        {
          addDisWordTuple(outputIndex + outputLength - n,
              inputIndex + inputLength - n, unkIndex + inputLength - n);
        }
      else if (type == 1)
        {
          addDisWordTuple(inputIndex + inputLength - n - 1,
              outputIndex + outputLength - n, unkIndex + outputLength - n);
        }
      else if (type == 3)
        {
          addDisWordTuple(outputIndex + outputLength - n - 1,
              inputIndex + inputLength - n, unkIndex + inputLength - n);
        }
    }
  return ngramNumber;
}

int
NgramWordTranslationDataSet::readText(ioFile* iof)
{
  // Cannot read text for word align models
  if (type == 4 || type == 5)
    {
      cerr << "ERROR: dwt with type " << type
          << " can not be used with text files" << endl;
      exit(1);
    }
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
NgramWordTranslationDataSet::resamplingText(ioFile* iof, int totalLineNumber,
    int resamplingLineNumber)
{
  // Cannot read text for word align models
  if (type == 4 || type == 5)
    {
      cerr << "ERROR: dwt with type " << type
          << " can not be used with text file" << endl;
      exit(1);
    }
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
NgramWordTranslationDataSet::createTensor()
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
NgramWordTranslationDataSet::readTextNgram(ioFile* iof)
{
  cerr << "ERROR: readTextNgram is called with NgramWordTranslationDataSet"
      << endl;
  exit(1);
}

int
NgramWordTranslationDataSet::readCoBiNgram(ioFile* iof)
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
NgramWordTranslationDataSet::writeReBiNgram(ioFile* iof)
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
NgramWordTranslationDataSet::addDisWordTuple(int* srcIndex, int* desIndex,
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
tupleCompare(const void *ngram1, const void *ngram2)
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
NgramWordTranslationDataSet::sortNgram()
{
  qsort((void*) data, (size_t) ngramNumber, (nm + 3) * sizeof(unsigned int),
      tupleCompare);

}
void
NgramWordTranslationDataSet::shuffle(int times)
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

int
NgramWordTranslationDataSet::writeReBiNgram()
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
  return 1;
}

float
NgramWordTranslationDataSet::computePerplexity()
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
NgramWordTranslationDataSet::addLine(string line)
{
  cerr
      << "ERROR: addLine(string line) is called with NgramWordTranslationDataSet"
      << endl;
  return 1;
}
