/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram neural network ranking language model
 *******************************************************************/
#include "mainModel.H"

NgramRankModel::NgramRankModel()
{
}

NgramRankModel::~NgramRankModel()
{
  delete baseNetwork;
  delete inputVoc;
  delete outputVoc;
  delete dataSet;
}

void
NgramRankModel::allocation()
{
  otl = new outils();
  otl->sgenrand(time(NULL));
  hiddenStep = 1;
  if (nonLinearType == TANH)
    {
      hiddenStep = 2;
    }
  else if (nonLinearType == SIGM)
    {
      hiddenStep = 2;
    }

  baseNetwork = new Sequential(hiddenLayerSizeArray.length * hiddenStep);
  outputNetwork = NULL;
  int i;
  if (name == RANKOVN)
    {
      baseNetwork->lkt = new LookupTable(inputVoc->wordNumber, dimensionSize,
          n, blockSize, 1, otl);
    }
  else if (name == RANKCN)
    {
      baseNetwork->lkt = new LookupTable(inputVoc->wordNumber, dimensionSize,
          n, blockSize, 0, otl);
    }
  Module* module;

  module = new Linear(n * dimensionSize, hiddenLayerSizeArray(0), blockSize,
      otl);
  baseNetwork->add(module);

  if (nonLinearType == TANH)
    {
      module = new Tanh(hiddenLayerSizeArray(0), blockSize); // non linear
      baseNetwork->add(module);
    }
  else if (nonLinearType == SIGM)
    {
      module = new Sigmoid(hiddenLayerSizeArray(0), blockSize); // non linear
      baseNetwork->add(module);
    }
  for (i = 1; i < hiddenLayerSizeArray.size[0]; i++)
    {
      module = new Linear(hiddenLayerSizeArray(i - 1), hiddenLayerSizeArray(i),
          blockSize, otl);
      baseNetwork->add(module);
      if (nonLinearType == TANH)
        {
          module = new Tanh(hiddenLayerSizeArray(i), blockSize); // non linear
          baseNetwork->add(module);
        }
      else if (nonLinearType == SIGM)
        {
          module = new Sigmoid(hiddenLayerSizeArray(i), blockSize); // non linear
          baseNetwork->add(module);
        }
    }
  module = new Linear(hiddenLayerSizeArray(i - 1), 1, blockSize, otl);
  baseNetwork->add(module);
  contextFeature = baseNetwork->output;
  gradContextFeature.resize(contextFeature);
  dataSet = new NgramRankDataSet(ngramType, n, BOS, inputVoc, outputVoc,
      mapIUnk, mapOUnk, BLOCK_NGRAM_NUMBER);
  delete otl;
}
NgramRankModel::NgramRankModel(string name, int ngramType,
    char* inputVocFileString, char* outputVocFileString, int mapIUnk,
    int mapOUnk, int BOS, int blockSize, int n, int dimensionSize,
    string nonLinearType, intTensor& hiddenLayerSizeArray)
{
  recurrent = 0;
  this->name = name;
  this->ngramType = ngramType;
  this->inputVoc = new SoulVocab(inputVocFileString);
  this->outputVoc = new SoulVocab(outputVocFileString);
  this->mapIUnk = mapIUnk;
  this->mapOUnk = mapOUnk;
  this->BOS = BOS;
  this->blockSize = blockSize;
  this->n = n;
  if (BOS > n - 1)
    {
      this->BOS = n - 1;
    }

  this->dimensionSize = dimensionSize;
  this->nonLinearType = nonLinearType;
  this->hiddenLayerSizeArray = hiddenLayerSizeArray;
  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  hiddenNumber = hiddenLayerSizeArray.length;
  allocation();
}

NgramRankModel::NgramRankModel(string name, int ngramType, int inputSize,
    int outputSize, int blockSize, int n, int dimensionSize,
    string nonLinearType, intTensor& hiddenLayerSizeArray)
{
  recurrent = 0;
  this->name = name;
  this->ngramType = ngramType;
  inputVoc = new SoulVocab();
  outputVoc = new SoulVocab();
  for (int i = 0; i < inputSize; i++)
    {
      stringstream out;
      out << i;
      inputVoc->add(out.str(), i);
    }
  for (int i = 0; i < outputSize; i++)
    {
      stringstream out;
      out << i;
      outputVoc->add(out.str(), i);
    }
  this->mapIUnk = 1;
  this->mapOUnk = 1;
  this->BOS = 1;
  this->blockSize = blockSize;
  this->n = n;
  this->dimensionSize = dimensionSize;
  this->nonLinearType = nonLinearType;
  this->hiddenLayerSizeArray = hiddenLayerSizeArray;
  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  hiddenNumber = hiddenLayerSizeArray.length;
  maxCodeWordLength = this->codeWord.size[1];
  outputNetworkNumber = outputNetworkSize.size[0];
  allocation();
}

void
NgramRankModel::firstTime()
{

}
void
NgramRankModel::firstTime(intTensor& context)
{

}

int
NgramRankModel::train(char* dataFileString, int maxExampleNumber,
    int iteration, string learningRateType, float learningRate,
    float learningRateDecay)
{
  ioFile dataIof;
  dataIof.takeReadFile(dataFileString);
  int ngramNumber;
  dataIof.readInt(ngramNumber);
  int N;
  dataIof.readInt(N);
  if (N < n)
    {
      cerr << "ERROR: N in data is wrong:" << N << " < " << n << endl;
      exit(1);
    }

  if (maxExampleNumber > ngramNumber || maxExampleNumber == 0)
    {
      maxExampleNumber = ngramNumber;
    }
  float currentLearningRate;
  int nstep;
  nstep = maxExampleNumber * (iteration - 1);
  intTensor readTensor(blockSize, N);
  intTensor ngram;
  ngram.sub(readTensor, 0, blockSize - 1, N - n, N - 1);
  ngram.t();
  int currentExampleNumber = 0;
  int percent = 1;
  float aPercent = maxExampleNumber * CONSTPRINT;
  float iPercent = aPercent * percent;
  int blockNumber = maxExampleNumber / blockSize;
  int remainingNumber = maxExampleNumber - blockSize * blockNumber;
  int i;
  cout << maxExampleNumber << " examples" << endl;
  for (i = 0; i < blockNumber; i++)
    {
      //Read one line and then train
      readTensor.readStrip(&dataIof); // read file n gram for word and context
      if (dataIof.getEOF())
        {
          break;
        }
      currentExampleNumber += blockSize;
      if (learningRateType == LEARNINGRATE_NORMAL)
        {
          currentLearningRate = learningRate / (1 + nstep * learningRateDecay);
        }
      else if (learningRateType == LEARNINGRATE_DOWN)
        {
          currentLearningRate = learningRate;
        }
      trainOne(ngram, currentLearningRate, blockSize);
      nstep += blockSize;
#if PRINT_DEBUG
      if (currentExampleNumber > iPercent)
        {
          percent++;
          iPercent = aPercent * percent;
          cout << (float) currentExampleNumber / maxExampleNumber << " ... "
              << flush;
        }
#endif
    }
  if (remainingNumber != 0 && !dataIof.getEOF())
    {
      ngram = 0;
      intTensor lastReadTensor(remainingNumber, N);
      lastReadTensor.readStrip(&dataIof);
      intTensor subReadTensor;
      subReadTensor.sub(readTensor, 0, remainingNumber - 1, 0, N - 1);
      subReadTensor.copy(lastReadTensor);
      if (!dataIof.getEOF())
        {
          if (learningRateType == LEARNINGRATE_NORMAL)
            {
              currentLearningRate = learningRate / (1 + nstep
                  * learningRateDecay);
            }
          else if (learningRateType == LEARNINGRATE_DOWN)
            {
              currentLearningRate = learningRate;
            }
          trainOne(ngram, currentLearningRate, remainingNumber);
        }
    }
#if PRINT_DEBUG
  cout << endl;
#endif
  return 1;
}

int
NgramRankModel::forwardProbability(intTensor& ngramTensor,
    floatTensor& probTensor)
{
  int ngramNumber = ngramTensor.size[0];
  intTensor ngram;
  int rBlockSize;
  int oBlockSize;
  int percent = 1;
  float aPercent = ngramNumber * CONSTPRINT;
  float iPercent = aPercent * percent;
  int maxBlockSize;
  rBlockSize = 0;
  int ngramId;

  do
    {
      ngramId = (rBlockSize + 1) * blockSize;
      maxBlockSize = blockSize;
      if (ngramId > ngramNumber)
        {
          maxBlockSize = ngramNumber - rBlockSize * blockSize;
          ngramId = ngramNumber;
        }
      ngram.sub(ngramTensor, rBlockSize * blockSize, ngramId - 1, 0, n - 1);
      ngram.t();
      baseNetwork->forward(ngram);
      for (oBlockSize = 0; oBlockSize < maxBlockSize; oBlockSize++)
        {
          if (oBlockSize % 2 == 0) //Possitive
            {
              probTensor(rBlockSize * blockSize + oBlockSize) = -contextFeature(
                  oBlockSize);
            }
          else//Negative
            {
              probTensor(rBlockSize * blockSize + oBlockSize)
                  = contextFeature(oBlockSize);
            }
        }
      rBlockSize++;
#if PRINT_DEBUG
      if (ngramId > iPercent)
        {
          percent++;
          iPercent = aPercent * percent;
          cout << (float) ngramId / ngramNumber << " ... " << flush;
        }
#endif
      //break;
    }
  while (ngramId < ngramNumber);

#if PRINT_DEBUG
  cout << endl;
#endif
  // for test
  //cout << "NgramRankModel::forwardProbability probTensor: " << endl;
  //probTensor.write();
  return 1;
}

void
NgramRankModel::trainOne(intTensor& ngram, float learningRate, int subBlockSize)
{
  intTensor localWord;
  intTensor idLocalWord(1, 1);
  baseNetwork->forward(ngram);
  gradContextFeature = 0;
  int rBlockSize;
  for (rBlockSize = 0; rBlockSize < subBlockSize / 2; rBlockSize++)
    {
      if (contextFeature(rBlockSize * 2) - contextFeature(rBlockSize * 2 + 1)
          < 1)
        {
          gradContextFeature(rBlockSize * 2) = -1;
          gradContextFeature(rBlockSize * 2 + 1) = 1;
        }
    }
  baseNetwork->backward(gradContextFeature);
  baseNetwork->updateParameters(learningRate);

}

void
NgramRankModel::read(ioFile* iof, int allocation, int blockSize)
{

  string readFormat;
  iof->readString(name);
  iof->readString(readFormat);
  iof->readInt(ngramType);
  inputVoc = new SoulVocab();
  outputVoc = new SoulVocab();
  iof->readInt(inputVoc->wordNumber);
  iof->readInt(outputVoc->wordNumber);
  iof->readInt(mapIUnk);
  iof->readInt(mapOUnk);
  iof->readInt(BOS);
  if (blockSize != 0)
    {
      this->blockSize = blockSize;
    }
  else
    {
      this->blockSize = DEFAULT_BLOCK_SIZE;
    }
  iof->readInt(n);
  iof->readInt(dimensionSize);
  iof->readInt(hiddenNumber);
  iof->readString(nonLinearType);
  hiddenLayerSizeArray.resize(hiddenNumber, 1);
  hiddenLayerSizeArray.read(iof);
  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  if (allocation)
    {
      this->allocation();
    }
  baseNetwork->read(iof);
  inputVoc->read(iof);
  outputVoc->read(iof);
}

void
NgramRankModel::write(ioFile* iof, int closeFile)
{
  iof->writeString(name);
  iof->writeString(iof->format);
  iof->writeInt(ngramType);
  iof->writeInt(inputVoc->wordNumber);
  iof->writeInt(outputVoc->wordNumber);
  iof->writeInt(mapIUnk);
  iof->writeInt(mapOUnk);
  iof->writeInt(BOS);
  iof->writeInt(n);
  iof->writeInt(dimensionSize);
  iof->writeInt(hiddenNumber);
  iof->writeString(nonLinearType);
  hiddenLayerSizeArray.write(iof);
  baseNetwork->write(iof);
  inputVoc->write(iof);
  outputVoc->write(iof);
  if (closeFile) {
	  iof->freeWriteFile();
  }
}

float
NgramRankModel::distance2(NeuralModel& anotherModel) {
	// TODO
	return 0;
}
