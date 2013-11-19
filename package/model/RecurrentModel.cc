/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for recurrent neural network language model
 Maybe we have bugs here :(
 *******************************************************************/
#include "mainModel.H"

RecurrentModel::RecurrentModel()
{
}

RecurrentModel::~RecurrentModel()
{
  delete baseNetwork;
  for (int idel = 0; idel < outputNetworkNumber; idel++)
    {
      delete outputNetwork[idel];
    }
  delete[] outputNetwork;
  delete inputVoc;
  delete outputVoc;
  delete dataSet;
}

void
RecurrentModel::allocation()
{
  recurrent = 1;
  mapIUnk = 1;//always
  mapOUnk = 1;//always
  if (name == COVR)
    {
      cont = 1;
    }
  else if (name == OVR)
    {
      cont = 0;
    }
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

  baseNetwork = new Sequential(hiddenLayerSizeArray.length * hiddenStep - 1);

  int i;

  baseNetwork->lkt = new LookupTable(inputVoc->wordNumber, dimensionSize,
      n - 1, blockSize, 1, otl);
  Module* module;

  if (name == COVR || name == OVR)
    {
      module = new RRLinear(dimensionSize, blockSize, n, nonLinearType, 1, otl);
    }
  if (name == COVR || name == OVR)
    {
      if (dimensionSize != hiddenLayerSizeArray(0))
        {
          cerr
              << "WARNING: first hidden layer size !=  projection dimension, use projection dimension"
              << endl;
        }
      hiddenLayerSizeArray(0) = dimensionSize;
    }
  baseNetwork->add(module);

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
  outputNetwork = new Module*[outputNetworkNumber];
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
  // For computing perplexity, with OVR, to be true, need forwardBlockSize = 1

  if (cont)
    {
      dataSet = new RecurrentDataSet(n, inputVoc, outputVoc, cont, 1,
          BLOCK_NGRAM_NUMBER);
    }
  else
    {
      dataSet = new RecurrentDataSet(n, inputVoc, outputVoc, cont, blockSize,
          BLOCK_NGRAM_NUMBER);
    }

  delete otl;
}

RecurrentModel::RecurrentModel(string name, char* inputVocFileString,
    char* outputVocFileString, int blockSize, int n, int dimensionSize,
    string nonLinearType, intTensor& hiddenLayerSizeArray,
    char* codeWordFileString, char* outputNetworkSizeFileString)
{
	//for test
	cout << "RecurrentModel::RecurrentModel first method" << endl;
  this->name = name;
  inputVoc = new SoulVocab(inputVocFileString);
  outputVoc = new SoulVocab(outputVocFileString);
  this->blockSize = blockSize;
  this->n = n;
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

RecurrentModel::RecurrentModel(string name, int inputSize, int outputSize,
    int blockSize, int n, int dimensionSize, string nonLinearType,
    intTensor& hiddenLayerSizeArray, intTensor& codeWord,
    intTensor& outputNetworkSize)
{
	//for test
	cout << "RecurrentModel::RecurrentModel second method" << endl;
  this->name = name;
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
  this->blockSize = blockSize;
  this->n = n;
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
RecurrentModel::firstTime()
{
  baseNetwork->modules[0]->firstTime = 1; // RRLinear
}

void
RecurrentModel::firstTime(intTensor& context)
{
  //If one of contexts has </s> as (n - 2)th word, we have to recompute, not just copy output
  if (!cont)
    {
      baseNetwork->modules[0]->iContext = 0;
      int check = 0;
      for (int i = 0; i < blockSize; i++)
        {
          if (context(n - 2, i) == inputVoc->es)
            {
              baseNetwork->modules[0]->iContext(i) = 1;
              check = 1;
            }
        }
      if (check)
        {
          //If one of contexts has </s> as (n - 2)th word, we have to recompute, not just copy output
          intTensor context1;
          context1.resize(context);
          context1 = inputVoc->ss;
          floatTensor currentOutput;
          currentOutput = baseNetwork->lkt->forward(context1);
          floatTensor ssFeature;
          ssFeature.sub(currentOutput, 0, dimensionSize - 1, 0, 0);
          baseNetwork->modules[0]->firstTime = 2;
          baseNetwork->modules[0]->forward(ssFeature);
        }
    }

}

int
RecurrentModel::train(char* dataFileString, int maxExampleNumber,
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
  int dataBlockSize;
  dataIof.readInt(dataBlockSize);
  if (dataBlockSize != blockSize)
    {
      cerr << "ERROR: blockSize (" << dataBlockSize << ") in data is wrong"
          << endl;
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
  intTensor selectReadTensor0;
  intTensor selectReadTensor1;
  intTensor context;
  intTensor word;
  context.sub(readTensor, 0, blockSize - 1, N - n, N - 2);
  context.t();
  word.select(readTensor, 1, N - 1);
  int currentExampleNumber = 0;
  int percent = 1;
  float aPercent = maxExampleNumber * CONSTPRINT;
  float iPercent = aPercent * percent;
  int blockNumber = maxExampleNumber / blockSize;
  int remainingNumber = maxExampleNumber - blockSize * blockNumber;
  int i, j;
  cout << maxExampleNumber << " examples" << endl;
  for (i = 0; i < blockNumber; i++)
    {
      //Read one line and then train
      if (i == 0)
        {
          readTensor.readStrip(&dataIof); // read file n gram for word and context
        }
      else
        {
          for (j = 0; j < N - 1; j++)
            {
              selectReadTensor0.select(readTensor, 1, j);
              selectReadTensor1.select(readTensor, 1, j + 1);
              selectReadTensor0.copy(selectReadTensor1);
            }
          selectReadTensor0.select(readTensor, 1, N - 1);
          selectReadTensor0.readStrip(&dataIof);
        }
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
      trainOne(context, word, currentLearningRate, blockSize);
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
          trainOne(context, word, currentLearningRate, remainingNumber);
        }
    }
#if PRINT_DEBUG
  cout << endl;
#endif
  return 1;
}

int
RecurrentModel::forwardProbability(intTensor& ngramTensor,
    floatTensor& probTensor)
{
	// for test
	//cout << "RecurrentModel::forwardProbability here" << endl;
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
  intTensor bContext(n - 1, blockSize);
  intTensor selectContext;
  intTensor selectBContext;
  intTensor context;
  context.sub(ngramTensor, 0, ngramNumber - 1, 0, n - 2);
  intTensor contextFlag;
  contextFlag.select(ngramTensor, 1, n + 2);
  intTensor word;
  word.select(ngramTensor, 1, n - 1);
  intTensor order;
  order.select(ngramTensor, 1, n + 1);
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

float
RecurrentModel::distance2(RecurrentModel& anotherModel) {
    float distSquared = 0;
    distSquared += baseNetwork->lkt->distance2(*(anotherModel.baseNetwork->lkt));
    for (int i = 0; i < hiddenLayerSizeArray.size[0]; i++) {
    	if (RRLinear* modules_casted=dynamic_cast<RRLinear*>(baseNetwork->modules[i])) {
    		RRLinear* another_modules_casted=dynamic_cast<RRLinear*>(anotherModel.baseNetwork->modules[i]);
    		distSquared+=modules_casted->distance2(*another_modules_casted);
    	}
    	if (Linear* modules_casted1=dynamic_cast<Linear*>(baseNetwork->modules[i])) {
			Linear* another_modules_casted1=dynamic_cast<Linear*>(anotherModel.baseNetwork->modules[i]);
			distSquared+=modules_casted1->distance2(*another_modules_casted1);
		}
    }
    for (int i = 0; i < outputNetworkNumber; i++) {
    	distSquared+=dynamic_cast<LinearSoftmax*>(outputNetwork[i])->distance2(*dynamic_cast<LinearSoftmax*>(anotherModel.outputNetwork[i]));
    }
    return distSquared;
}

void
RecurrentModel::read(ioFile* iof, int allocation, int blockSize)
{
	// for test
	//cout << "RecurrentModel::read here 10" << endl;
  string readFormat;
  iof->readString(name);
  iof->readString(readFormat);
  // for test
  //cout << "RecurrentModel::read here" << endl;
  inputVoc = new SoulVocab();
  outputVoc = new SoulVocab();
  // for test
  //cout << "RecurrentModel::read here 1" << endl;
  iof->readInt(inputVoc->wordNumber);
  iof->readInt(outputVoc->wordNumber);
  // for test
  //cout << "RecurrentModel::read here 2" << endl;
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
  iof->readInt(maxCodeWordLength);
  iof->readInt(outputNetworkNumber);
  // for test
  cout << "RecurrentModel::read here 3" << endl;
  codeWord.resize(outputVoc->wordNumber, maxCodeWordLength);
  outputNetworkSize.resize(outputNetworkNumber, 1);
  codeWord.read(iof);
  outputNetworkSize.read(iof);
  hiddenLayerSizeArray.resize(hiddenNumber, 1);
  hiddenLayerSizeArray.read(iof);
  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  // for test
  //cout << "RecurrentModel::read here 4" << endl;
  if (allocation)
    {
	  // for test
	  //cout << "RecurrentModel::read here 5" << endl;
      this->allocation();
    }
  // for test
  //cout << "RecurrentModel::read here 6" << endl;
  baseNetwork->read(iof);
  int i;
  // for test
  //cout << "RecurrentModel::read here 7" << endl;
  for (i = 0; i < outputNetworkSize.size[0]; i++)
    {
      outputNetwork[i]->read(iof);
    }
  // for test
  //cout << "RecurrentModel::read here 8" << endl;
  inputVoc->read(iof);
  // for test
  //cout << "RecurrentModel::read here 9" << endl;
  outputVoc->read(iof);
}
void
RecurrentModel::write(ioFile* iof, int closeFile)
{
  iof->writeString(name);
  iof->writeString(iof->format);
  iof->writeInt(inputVoc->wordNumber);
  iof->writeInt(outputVoc->wordNumber);
  iof->writeInt(n);
  iof->writeInt(dimensionSize);
  iof->writeInt(hiddenNumber);
  iof->writeString(nonLinearType);
  iof->writeInt(maxCodeWordLength);
  iof->writeInt(outputNetworkNumber);
  codeWord.write(iof);
  outputNetworkSize.write(iof);
  hiddenLayerSizeArray.write(iof);
  baseNetwork->write(iof);
  int i;
  for (i = 0; i < outputNetworkSize.size[0]; i++)
    {
      outputNetwork[i]->write(iof);
    }
  inputVoc->write(iof);
  outputVoc->write(iof);
  if (closeFile == 1) {
	  iof->freeWriteFile();
  }
}

