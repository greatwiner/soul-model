/*******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for recurrent data set.
 *******************************************************************/

#include "text.H"
RecurrentDataSet::RecurrentDataSet(int n, SoulVocab* inputVoc,
    SoulVocab* outputVoc, int cont, int blockSize, int maxNgramNumber)
{
  this->n = n;
  this->BOS = -1;
  this->inputVoc = inputVoc;
  this->outputVoc = outputVoc;
  this->cont = cont;
  this->blockSize = blockSize;
  this->mapIUnk = 1;
  this->mapOUnk = 1;
  data = NULL;
  ngramNumber = 0;
  dataTensor.resize(1, 1);
  try
    {
      data = new int[maxNgramNumber * (n + 3)]; //Two ints code information, the last is ID_END_NGRAM
    }
  catch (bad_alloc& ba)
    {
      cerr << "RecurrentDataSet bad_alloc caught: " << ba.what() << endl;
      exit(1);
    }
}

int
RecurrentDataSet::addLine(string line)
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
    }

  if (length == BOS - 1) // The line have no ngram, don't do anything
    {
      return 0;
    }
  if (ngramNumber >= 1 && cont)
    {
      for (j = 0; j < n - 1; j++)
        {
          data[ngramNumber * (n + 3) + j] = data[(ngramNumber - 1) * (n + 3)
              + j + 1];
        }
    }
  else
    {
      for (j = 0; j < n - 2; j++)
        {
          data[ngramNumber * (n + 3) + j] = inputVoc->ss;
        }
      data[ngramNumber * (n + 3) + n - 2] = inputVoc->es;
    }
  data[ngramNumber * (n + 3) + n - 1] = outputIndex[i];
  data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
  //Normaly, is the order of this ngram in file
  data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
  //The value to find the indentical context and ...
  data[ngramNumber * (n + 3) + n + 2] = 0;
  ngramNumber++;
  int lastId = n - 1;
  if (lastId > length)
    {
      lastId = length;
    }
  for (i = 1; i < lastId; i++)
    {
      for (j = 0; j < n - 1; j++)
        {
          data[ngramNumber * (n + 3) + j] = data[(ngramNumber - 1) * (n + 3)
              + j + 1];
        }
      data[ngramNumber * (n + 3) + n - 1] = outputIndex[i];
      data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
      //Normaly, is the order of this ngram in file
      data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
      //The value to find the indentical context and ...
      data[ngramNumber * (n + 3) + n + 2] = 0;
      ngramNumber++;
    }

  for (i = 0; i <= length - n; i++)
    {
      use = 1;
      for (j = 0; j < n - 1; j++)
        {
          if (inputIndex[i + j] == ID_UNK)
            {
              use = 0;
              break;
            }
          data[ngramNumber * (n + 3) + j] = inputIndex[i + j];
        }
      if (outputIndex[i + n - 1] == ID_UNK)
        {
          use = 0;
        }
      else
        {
          data[ngramNumber * (n + 3) + n - 1] = outputIndex[i + n - 1];
        }
      if (use)
        {
          data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
          //Normaly, is the order of this ngram in file
          data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
          //The value to find the indentical context and ...
          data[ngramNumber * (n + 3) + n + 2] = 0;
          ngramNumber++;
        }
    }
  return 1;
}

int
RecurrentDataSet::resamplingSentence(int totalLineNumber,
    int resamplingLineNumber, int* resamplingLineId)
{
	// for test
	//cout << "RecurrentDataSet::resamplingSentence here" << endl;
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
      if (cont)
        {
          int chosenPos;
          int i;
          chosenPos = rand() % totalLineNumber;
          resamplingLineId[0] = chosenPos;
          for (i = 1; i < resamplingLineNumber; i++)
            {
              resamplingLineId[i] = (resamplingLineId[0] + i) % totalLineNumber;
            }
        }
      else
        {
    	  // for test
    	  //cout << "RecurrentDataSet::resamplingSentence here 1" << endl;
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
        }
      sort(resamplingLineId, resamplingLineId + resamplingLineNumber);
      return 1;
    }
}

int
RecurrentDataSet::readText(ioFile* iof)
{

  string line;
  int readLineNumber = 0;
  int currentId = 0;
  while (!iof->getEOF())
    {
      if (iof->getLine(line))
        {
          if (!checkBlankString(line))
            {
              line = line + " " + ES;
              addLine(line);
            }
          currentId++;
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
RecurrentDataSet::resamplingText(ioFile* iof, int totalLineNumber,
    int resamplingLineNumber)
{
	// for test
	//cout << "RecurrentDataSet::resamplingText here" << endl;
  int* resamplingLineId = new int[resamplingLineNumber];
  // for test
  //cout << "RecurrentDataSet::resamplingText here 1" << endl;
  resamplingSentence(totalLineNumber, resamplingLineNumber, resamplingLineId);
  // for test
  //cout << "RecurrentDataSet::resamplingText here 2" << endl;

  string line;
  string headline;
  headline = "";
  int readLineNumber = 0;
  int currentId = 0;
  while (!iof->getEOF())
    {
      if (iof->getLine(line))
        {
          if (readLineNumber == resamplingLineId[currentId])
            {
              if (!checkBlankString(line))
                {
                  line = headline + line + " " + ES;
                  addLine(line);
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
RecurrentDataSet::createTensor()
{
	// for test
	//cout << "RecurrentDataSet::createTensor here" << endl;
  intTensor pos;
  pos.resize(blockSize, 1);
  intTensor dis;
  dis.resize(blockSize, 1);
  // for test
  //cout << "RecurrentDataSet::createTensor blockSize: " << blockSize << endl;

  int outMBlock = findPos(pos, dis) / (n + 3);
  // for test
  //cout << "RecurrentDataSet::createTensor here 4" << endl;
  int outNgramNumber = blockSize * outMBlock;
  int rNBlock;
  int rBlockSize;
  intTensor ssArray;
  ssArray.resize(n + 2, 1);
  // for test
  //cout << "RecurrentDataSet::createTensor here 5" << endl;
  ssArray = inputVoc->ss;
  // for test
  //cout << "RecurrentDataSet::createTensor here 6" << endl;
  ssArray(n - 1) = SIGN_NOT_WORD;
  ssArray(n) = ID_END_NGRAM;
  ssArray(n + 1) = SIGN_NOT_WORD;
  // for test
  //cout << "RecurrentDataSet::createTensor here 1" << endl;

  dataTensor.resize(outNgramNumber, n + 3);
  int rN;
  for (rNBlock = 0; rNBlock < outMBlock; rNBlock++)
    {
      for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++)
        {
          if (rNBlock * (n + 3) < dis(rBlockSize))
            {
              for (rN = 0; rN < n + 2; rN++)
                {
                  dataTensor(rBlockSize + rNBlock * blockSize, rN) = data[pos(
                      rBlockSize) + rNBlock * (n + 3) + rN];
                }
            }
          else
            {
              for (rN = 0; rN < n + 2; rN++)
                {
                  dataTensor(rBlockSize + rNBlock * blockSize, rN)
                      = ssArray(rN);
                }
            }
          dataTensor(rBlockSize + rNBlock * blockSize, n + 2) = rBlockSize
              + rNBlock * blockSize + 1;
        }
    }
  // for test
  //cout << "RecurrentDataSet::createTensor here 2" << endl;
  probTensor.resize(ngramNumber, 1);
  // for test
  //cout << "RecurrentDataSet::createTensor here 3" << endl;
  //cout << "RecurrentDataSet::createTensor output of all: " << endl;
  //dataTensor.write();
  return dataTensor;
}

int
RecurrentDataSet::readTextNgram(ioFile* iof)
{
  string line;
  int readLineNumber = 0;
  while (!iof->getEOF())
    {
      if (iof->getLine(line))
        {
          if (!checkBlankString(line))
            {
              addLine(line);
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

void
RecurrentDataSet::writeReBiNgram(ioFile* iof)
{
  intTensor pos;
  pos.resize(blockSize, 1);
  intTensor dis;
  dis.resize(blockSize, 1);
  int outMBlock = findPos(pos, dis);
  iof->writeInt(blockSize * outMBlock / (n + 3));
  iof->writeInt(n);
  iof->writeInt(blockSize);
  int rNBlock = 0;//??
  int rBlockSize;
  intTensor ssArray;
  ssArray.resize(n, 1);
  ssArray = inputVoc->ss;
  ssArray(n - 1) = SIGN_NOT_WORD;
  //Write first 'ngram'
  for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++)
    {
      if (rNBlock < dis(rBlockSize))
        {
          iof->writeIntArray(data + pos(rBlockSize), n);
        }
      else
        {
          iof->writeIntArray(ssArray.data, n);
        }
    }
  //Now write only word or </s>
  for (rNBlock = n + 3; rNBlock < outMBlock; rNBlock += n + 3)
    {
      for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++)
        {
          if (rNBlock < dis(rBlockSize))
            {
              iof->writeIntArray(data + pos(rBlockSize) + rNBlock + n - 1, 1);
            }
          else
            {
              iof->writeIntArray(ssArray.data + n - 1, 1);
            }
        }
    }
}

int
RecurrentDataSet::findPos(intTensor& pos, intTensor& dis)
{
	// for test
	//cout << "RecurrentDataSet::findPos here" << endl;
	//cout << "RecurrentDataSet::findPos pos: " << endl;
	//pos.write();
	//cout << "RecurrentDataSet::findPos dis" << endl;
	//dis.write();
  pos(0) = 0;
  int rBlockSize;
  int dBlockSize = ngramNumber / blockSize;
  // for test
  //cout << "RecurrentDataSet::findPos ngramNumber: " << ngramNumber << endl;
  //cout << "RecurrentDataSet::findPos dBlockSize: " << dBlockSize << endl;
  if (dBlockSize * blockSize < ngramNumber)
    {
      dBlockSize++;
    }
  // for test
  //cout << "RecurrentDataSet::findPos here 1" << endl;
  int max = 0;
  int maxPos = (ngramNumber - 1) * (n + 3);
  // for test
  //cout << "RecurrentDataSet::findPos here 2" << endl;
  for (rBlockSize = 1; rBlockSize < blockSize; rBlockSize++)
    {
	  // for test
	  //cout << "RecurrentDataSet::findPos rBlockSize: " << rBlockSize << endl;
      pos(rBlockSize) = pos(rBlockSize - 1) + (dBlockSize - 1) * (n + 3);
      // for test
      //cout << "RecurrentDataSet::findPos here 5" << endl;
      if (pos(rBlockSize) > maxPos)
        {
    	  // for test
    	  //cout << "RecurrentDataSet::findPos here 6" << endl;
          pos(rBlockSize) = maxPos;
        }
      else
        {
    	  // for test
    	  //cout << "RecurrentDataSet::findPos here 7" << endl;
    	  //cout << "RecurrentDataSet::findPos output->es: " << outputVoc->es << endl;
          while (data[pos(rBlockSize) + n - 1] != outputVoc->es)
            {
        	  // for test
        	  //cout << "RecurrentDataSet::findPos continue bouque" << endl;
        	  //cout << data[pos(rBlockSize) + n - 1];
              pos(rBlockSize) += n + 3;
            }
          pos(rBlockSize) += n + 3;
        }
      // for test
      //cout << "RecurrentDataSet::findPos here 8" << endl;
      dis(rBlockSize - 1) = pos(rBlockSize) - pos(rBlockSize - 1);
      // for test
      //cout << "RecurrentDataSet::findPos here 9" << endl;
      if (max < dis(rBlockSize - 1))
        {
    	  // for test
    	  //cout << "RecurrentDataSet::findPos here 10" << endl;
          max = dis(rBlockSize - 1);
        }

    }
  // for test
  //cout << "RecurrentDataSet::findPos here 3" << endl;
  dis(blockSize - 1) = ngramNumber * (n + 3) - pos(blockSize - 1);
  if (max < dis(blockSize - 1))
    {
      max = dis(blockSize - 1);
    }
  // for test
  //cout << "RecurrentDataSet::findPos here 4" << endl;
  return max;
}

int
RecurrentDataSet::readCoBiNgram(ioFile* iof)
{
  cerr << "ERROR: readCoBiNgram is called with RecurrentDataSet" << endl;
  exit(1);
}

float
RecurrentDataSet::computePerplexity()
{
  perplexity = 0;
  for (int i = 0; i < probTensor.length; i++)
    {
      perplexity += log(probTensor(i));
    }
  // for test
  cout << "RecurrentDataSet::computePerplexity perplexity: " << perplexity << endl;
  perplexity = exp(-perplexity / ngramNumber);
  return perplexity;
}

int
RecurrentDataSet::addLine(ioFile* iof)
{
  string line;
  if (iof->getLine(line))
    {
      if (!checkBlankString(line))
        {
          line = line + " " + ES;
          return addLine(line);
        }
    }
  return 0;
}

