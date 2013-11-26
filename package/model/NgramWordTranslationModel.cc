/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram neural network word factored translation model
 *******************************************************************/
#include "mainModel.H"

NgramWordTranslationModel::NgramWordTranslationModel()
{
}

NgramWordTranslationModel::~NgramWordTranslationModel()
{
	// for test
	//cout << "NgramWordTranslationModel::~NgramWordTranslationModel here" << endl;
  delete baseNetwork;
  // for test
  //cout << "NgramWordTranslationModel::~NgramWordTranslationModel here 1" << endl;
  for (int idel = 0; idel < outputNetworkNumber; idel++)
    {
	  // for test
	  //cout << "NgramWordTranslationModel::~NgramWordTranslationModel idel: " << idel << endl;
      delete outputNetwork[idel];
    }
  // for test
  //cout << "NgramWordTranslationModel::~NgramWordTranslationModel here 2" << endl;
  delete[] outputNetwork;
  // for test
  //cout << "NgramWordTranslationModel::~NgramWordTranslationModel here 3" << endl;
  delete inputVoc;
  // for test
  //cout << "NgramWordTranslationModel::~NgramWordTranslationModel here 4" << endl;
  delete outputVoc;
  // for test
  //cout << "NgramWordTranslationModel::~NgramWordTranslationModel here 5" << endl;
  delete dataSet;
  // for test
  //cout << "NgramWordTranslationModel::~NgramWordTranslationModel here 6" << endl;
}

void
NgramWordTranslationModel::allocation()
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

  int i;
  baseNetwork->lkt = new LookupTable(inputVoc->wordNumber, dimensionSize,
      nm - 1, blockSize, 1, otl);

  Module* module;
  if (name == WTOVN)
    {
      module = new Linear((nm - 1) * dimensionSize, hiddenLayerSizeArray(0),
          blockSize, otl);
    }
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
  probabilityOne.resize(blockSize, 1);
  int outputNetworkNumber = outputNetworkSize.size[0];
  outputNetwork = new ProbOutput*[outputNetworkNumber];
  LinearSoftmax* sl = new LinearSoftmax(hiddenLayerSize, outputNetworkSize(0),
      blockSize, otl);
  outputNetwork[0] = sl;
  for (i = 1; i < outputNetworkNumber; i++)
    {
      sl = new LinearSoftmax(hiddenLayerSize, outputNetworkSize(i), 1, otl);
      outputNetwork[i] = sl;
    }
  doneForward.resize(outputNetworkNumber, 1);
  contextFeature = baseNetwork->output;
  gradContextFeature.resize(contextFeature);
  localCodeWord.resize(blockSize, maxCodeWordLength);
  dataSet = new NgramWordTranslationDataSet(ngramType, n, BOS, inputVoc,
      outputVoc, mapIUnk, mapOUnk, BLOCK_NGRAM_NUMBER);

  delete otl;
}
NgramWordTranslationModel::NgramWordTranslationModel(string name,
    int ngramType, char* inputVocFileString, char* outputVocFileString,
    int mapIUnk, int mapOUnk, int BOS, int blockSize, int n, int dimensionSize,
    string nonLinearType, intTensor& hiddenLayerSizeArray,
    char* codeWordFileString, char* outputNetworkSizeFileString)
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
  nm = n * 2;
  if (BOS > n)
    {
      this->BOS = n;
    }

  this->dimensionSize = dimensionSize;
  this->nonLinearType = nonLinearType;
  ioFile readIof;
  if (!strcmp(codeWordFileString, "xxx"))
    {
      codeWord.resize(outputVoc->wordNumber, 2);
      codeWord = 0;
      for (int wordIndex = 0; wordIndex < outputVoc->wordNumber; wordIndex++)
        {
          codeWord(wordIndex, 1) = wordIndex;
        }
    }
  else
    {
      readIof.takeReadFile(codeWordFileString);
      codeWord.read(&readIof);
    }
  if (!strcmp(outputNetworkSizeFileString, "xxx"))
    {
      outputNetworkSize.resize(1, 1);
      outputNetworkSize(0) = outputVoc->wordNumber;
    }
  else
    {
      readIof.takeReadFile(outputNetworkSizeFileString);
      outputNetworkSize.read(&readIof);
    }
  this->hiddenLayerSizeArray.resize(hiddenLayerSizeArray);
  this->hiddenLayerSizeArray.copy(hiddenLayerSizeArray);
  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  hiddenNumber = hiddenLayerSizeArray.length;
  maxCodeWordLength = this->codeWord.size[1];
  outputNetworkNumber = outputNetworkSize.size[0];
  allocation();
}

NgramWordTranslationModel::NgramWordTranslationModel(string name,
    int ngramType, int inputSize, int outputSize, int blockSize, int n,
    int dimensionSize, string nonLinearType, intTensor& hiddenLayerSizeArray,
    intTensor& codeWord, intTensor& outputNetworkSize)
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
  this->BOS = n;
  this->blockSize = blockSize;
  this->n = n;
  nm = n * 2;
  this->dimensionSize = dimensionSize;
  this->nonLinearType = nonLinearType;
  this->hiddenLayerSizeArray = hiddenLayerSizeArray;
  this->codeWord = codeWord;
  this->outputNetworkSize = outputNetworkSize;
  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  hiddenNumber = hiddenLayerSizeArray.length;
  maxCodeWordLength = this->codeWord.size[1];
  outputNetworkNumber = outputNetworkSize.size[0];
  allocation();
}

void
NgramWordTranslationModel::firstTime()
{

}
void
NgramWordTranslationModel::firstTime(intTensor& context)
{

}

int
NgramWordTranslationModel::train(char* dataFileString, int maxExampleNumber,
    int iteration, string learningRateType, float learningRate,
    float learningRateDecay)
{
  firstTime();
  ioFile dataIof;
  dataIof.takeReadFile(dataFileString);
  int ngramNumber;
  dataIof.readInt(ngramNumber);
  int N;
  dataIof.readInt(N);
  if (N < nm)
    {
      cerr << "ERROR: N in data is wrong:" << N << " < " << nm << endl;
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
  intTensor context;
  intTensor word;
  floatTensor coefTensor(blockSize, 1);
  context.sub(readTensor, 0, blockSize - 1, N - nm, N - 2);
  context.t();
  word.select(readTensor, 1, N - 1);
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
      this->readStripInt(dataIof, readTensor, coefTensor); // read file n gram for word and context
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
      trainOne(context, word, coefTensor, currentLearningRate, blockSize);
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
      context = 0;
      word = SIGN_NOT_WORD;
      intTensor lastReadTensor(remainingNumber, N);
      lastReadTensor.readStrip(&dataIof);
      this->readStripInt(dataIof, lastReadTensor, coefTensor);
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
          trainOne(context, word, coefTensor, currentLearningRate, remainingNumber);
        }
    }
#if PRINT_DEBUG
  cout << endl;
#endif
  return 1;
}

int
NgramWordTranslationModel::forwardProbability(intTensor& ngramTensor,
    floatTensor& probTensor)
{
  int bkBlockSize = blockSize;
  if (cont && recurrent)
    {
      changeBlockSize(1);
    }
  firstTime();
  int localWord;
  int idParent;
  int i;
  float localProb;
  int idWord;
  int ngramNumber = ngramTensor.size[0];
  intTensor oneLocalCodeWord;
  intTensor bContext(nm - 1, blockSize);
  intTensor selectContext;
  intTensor selectBContext;
  intTensor context;
  context.sub(ngramTensor, 0, ngramNumber - 1, 0, nm - 2);
  intTensor contextFlag;
  contextFlag.select(ngramTensor, 1, nm + 2);
  intTensor word;
  word.select(ngramTensor, 1, nm - 1);
  intTensor order;
  order.select(ngramTensor, 1, nm + 1);
  int ngramId = 0;
  int ngramId2 = 0;
  int rBlockSize;
  int nextId;
  int percent = 1;
  float aPercent = ngramNumber * CONSTPRINT;
  float iPercent = aPercent * percent;
  bContext = 0;
  do
    {
      ngramId2 = ngramId;
      rBlockSize = 0;

      while (rBlockSize < blockSize && ngramId < ngramNumber)
        {
          selectBContext.select(bContext, 1, rBlockSize);
          selectContext.select(context, 0, ngramId);
          selectBContext.copy(selectContext);
          ngramId = contextFlag(ngramId);
          rBlockSize++;
        }
      rBlockSize = 0;
      firstTime(bContext);
      baseNetwork->forward(bContext);
      mainProb = outputNetwork[0]->forward(contextFeature);
      while (rBlockSize < blockSize && ngramId2 < ngramNumber)
        {
          doneForward = 0;
          nextId = contextFlag(ngramId2);
          for (; ngramId2 < nextId; ngramId2++)
            {
              if (order(ngramId2) != SIGN_NOT_WORD)
                {
                  intTensor oneLocalCodeWord;
                  idWord = word(ngramId2);
                  oneLocalCodeWord.select(codeWord, 0, idWord);
                  localWord = oneLocalCodeWord(1);
                  localProb = mainProb(localWord, rBlockSize);
                  for (i = 2; i < maxCodeWordLength; i += 2)
                    {
                      if (oneLocalCodeWord(i) == SIGN_NOT_WORD)
                        {
                          break;
                        }
                      localWord = oneLocalCodeWord(i + 1);
                      idParent = oneLocalCodeWord(i);
                      if (!doneForward(idParent))
                        {
                          selectContextFeature.select(contextFeature, 1,
                              rBlockSize);
                          outputNetwork[idParent]->forward(selectContextFeature);
                          doneForward(idParent) = 1;
                        }
                      localProb *= outputNetwork[idParent]->output(localWord);
                    }
                  probTensor(order(ngramId2)) = localProb;
                }
            }
          rBlockSize++;
        }
#if PRINT_DEBUG
      if (ngramId > iPercent)
        {
          percent++;
          iPercent = aPercent * percent;
          cout << (float) ngramId / ngramNumber << " ... " << flush;
        }
#endif
    }
  while (ngramId < ngramNumber);
#if PRINT_DEBUG
  cout << endl;
#endif
  if (cont && recurrent)
    {
      changeBlockSize(bkBlockSize);
    }
  return 1;
}

void
NgramWordTranslationModel::read(ioFile* iof, int allocation, int blockSize)
{
  string readFormat;

  // for test
  //cout << "NgramWordTranslationModel::read here " << endl;
  iof->readString(name);
  iof->readString(readFormat);
  iof->readInt(ngramType);
  // for test
  //cout << "NgramWordTranslationModel::read here 1" << endl;
  //cout << "NgramWordTranslationModel::read name: " << name << endl;
  //cout << "NgramWordTranslationModel::read readFormat: " << readFormat << endl;
  //cout << "NgramWordTranslationModel::read ngramType: " << ngramType << endl;
  inputVoc = new SoulVocab();
  outputVoc = new SoulVocab();
  iof->readInt(inputVoc->wordNumber);
  iof->readInt(outputVoc->wordNumber);
  iof->readInt(mapIUnk);
  iof->readInt(mapOUnk);
  iof->readInt(BOS);
  // for test
  //cout << "NgramWordTranslationModel::read here 2" << endl;
  //cout << "NgramWordTranslationModel::read inputVoc->wordNumber: " << inputVoc->wordNumber << endl;
  //cout << "NgramWordTranslationModel::read outputVoc->wordNumber: " << outputVoc->wordNumber << endl;
  if (blockSize != 0)
    {
      this->blockSize = blockSize;
    }
  else
    {
      this->blockSize = DEFAULT_BLOCK_SIZE;
    }
  iof->readInt(n);
  // for test
  //cout << "NgramWordTranslationModel::read here 3" << endl;
  //cout << "NgramWordTranslationModel::read n: " << n << endl;
  nm = n * 2;
  iof->readInt(dimensionSize);
  iof->readInt(hiddenNumber);
  iof->readString(nonLinearType);
  iof->readInt(maxCodeWordLength);
  iof->readInt(outputNetworkNumber);
  // for test
  //cout << "NgramWordTranslationModel::read here 4" << endl;
  codeWord.resize(outputVoc->wordNumber, maxCodeWordLength);
  outputNetworkSize.resize(outputNetworkNumber, 1);
  codeWord.read(iof);
  // for test
  //cout << "NgramWordTranslationModel::read here 5" << endl;
  outputNetworkSize.read(iof);
  // for test
  //cout << "NgramWordTranslationModel::read here 6" << endl;
  hiddenLayerSizeArray.resize(hiddenNumber, 1);
  hiddenLayerSizeArray.read(iof);
  // for test
  //cout << "NgramWordTranslationModel::read here 7" << endl;
  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  if (allocation)
    {
      this->allocation();
    }
  // for test
  //cout << "NgramWordTranslationModel::read here 71" << endl;
  baseNetwork->read(iof);
  // for test
  //cout << "NgramWordTranslationModel::read here 8" << endl;
  // for test
  //cout << "NgramWordTranslationModel::read lkt name: " << baseNetwork->lkt->name << endl;
  //cout << "NgramWordTranslationModel::read first hidden layer name: " << baseNetwork->modules[0]->name << endl;
  int i;
  for (i = 0; i < outputNetworkSize.size[0]; i++)
    {
      outputNetwork[i]->read(iof);
      // for test
      //cout << "NgramWordTranslationModel::read here 9" << endl;
      //cout << "NgramWordTranslationModel::read outputNetwork " << i << " name: " << outputNetwork[i]->name << endl;
    }
  inputVoc->read(iof);
  // for test
  //cout << "NgramWordTranslationModel::read here 10" << endl;
  outputVoc->read(iof);
  // for test
  //cout << "NgramWordTranslationModel::read here 11" << endl;
  // for test
  //cout << "NgramWordTranslationModel::read reference: " << outputVoc->table[0]->next->next << " index: " << outputVoc->table[0]->next->next->index << " word: " << outputVoc->table[0]->next->next->word << endl;
}

void
NgramWordTranslationModel::write(ioFile* iof, int closeFile)
{
	// for test
	//cout << "NgramWordTranslationModel::write here" << endl;
  iof->writeString(name);
  // for test
  //cout << "NgramWordTranslationModel::write name: " << name << endl;
  iof->writeString(iof->format);
  // for test
  //cout << "NgramWordTranslationModel::write iof->format: " << iof->format << endl;

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
  iof->writeInt(maxCodeWordLength);
  iof->writeInt(outputNetworkNumber);
  // for test
  //cout << "NgramWordTranslationModel::write here 1" << endl;
  codeWord.write(iof);
  // for test
  //cout << "NgramWordTranslationModel::write here 2" << endl;
  outputNetworkSize.write(iof);
  // for test
  //cout << "NgramWordTranslationModel::write here 3" << endl;
  hiddenLayerSizeArray.write(iof);
  // for test
  //cout << "NgramWordTranslationModel::write here 4" << endl;
  baseNetwork->write(iof);
  // for test
  //cout << "NgramWordTranslationModel::write here 5" << endl;
  int i;
  for (i = 0; i < outputNetworkSize.size[0]; i++)
    {
      outputNetwork[i]->write(iof);
      // for test
      //cout << "NgramWordTranslationModel::write i: " << i << endl;
    }
  // for test
  //cout << "NgramWordTranslationModel::write here 6" << endl;
  inputVoc->write(iof);
  // for test
  //cout << "NgramWordTranslationModel::write here 7" << endl;
  outputVoc->write(iof);
  // for test
  //cout << "NgramWordTranslationModel::write here 8" << endl;
  //iof->freeWriteFile();
  // for test
  //cout << "NgramWordTranslationModel::write here 9" << endl;
  if (closeFile == 1) {
	  iof->freeWriteFile();
  }
}

float
NgramWordTranslationModel::distance2(NeuralModel& anotherModel) {
	// TODO
	return 0;
}
