/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for n-gram neural network language model
 *******************************************************************/
#include "mainModel.H"

NgramModel::NgramModel()
{
}

NgramModel::~NgramModel()
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
NgramModel::allocation()
{
  recurrent = 0;
  otl = new outils();
  otl->sgenrand(time(NULL) + getpid());
  // hiddenStep = 1 if linear, 2 if non linear
  hiddenStep = 1;
  if (nonLinearType == TANH)
    {
      hiddenStep = 2;
    }
  else if (nonLinearType == SIGM)
    {
      hiddenStep = 2;
    }

  if (name == OVNB) {
  	  baseNetwork = new Sequential_Bayes(hiddenLayerSizeArray.length * hiddenStep);
  }
  else {
  	  baseNetwork = new Sequential(hiddenLayerSizeArray.length * hiddenStep);
  }

  int i;
  // Lookup table with classical or one vector initialization
  if (name == CN || name == LBL)
    {
      baseNetwork->lkt = new LookupTable(inputVoc->wordNumber, dimensionSize,
          n - 1, blockSize, 0, otl);
    }
  else if (name == OVN || name == OVNB || name == ROVN || name == MAXOVN)
    {
	    if (name == OVNB) {
	    	baseNetwork->lkt = new LookupTable_Bayes(inputVoc->wordNumber, dimensionSize,
	  				  n - 1, blockSize, 1, otl);
	    }
	    else {
	    	baseNetwork->lkt = new LookupTable(inputVoc->wordNumber, dimensionSize,
	  				  n - 1, blockSize, 1, otl);
	    }
    }
  Module* module;
  // First module depends on the type of model
  if (name == ROVN)
    {
      module = new RLinear(dimensionSize, blockSize, n, nonLinearType, 1, otl);

      if (dimensionSize != hiddenLayerSizeArray(0))
        {
          cerr
              << "WARNING: first hidden layer size !=  projection dimension, use projection dimension"
              << endl;
        }
      baseNetwork->add(module);
    }
  else if (name == MAXOVN)
    {
      module = new MaxLinear(n - 1, dimensionSize, hiddenLayerSizeArray(0),
          blockSize, otl);
      baseNetwork->add(module);
    }
  else if (name == LBL)
    {
      module = new Linear((n - 1) * dimensionSize, hiddenLayerSizeArray(0),
          blockSize, otl);
      baseNetwork->add(module);

      if (dimensionSize
          != hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1))
        {
          cerr
              << "WARNING: last hidden layer size !=  projection dimension, use projection dimension"
              << endl;
        }
      hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1) = dimensionSize;
    }
  // CN, OVN, OVNB
  else
    {
	    if (name == OVNB) {
	    	module = new Linear_Bayes((n-1)*dimensionSize, hiddenLayerSizeArray(0),
	      	          blockSize, otl);
	    	baseNetwork->add(module);
		}
		else {
			module = new Linear((n - 1) * dimensionSize, hiddenLayerSizeArray(0),
	      			  blockSize, otl);
			baseNetwork->add(module);
		}
    }
  // Add non linear activation
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
  // Add several hidden layers with linear or non linear activation
  for (i = 1; i < hiddenLayerSizeArray.size[0]; i++)
    {
	  if (name == OVNB) {
		  module = new Linear_Bayes(hiddenLayerSizeArray(i - 1), hiddenLayerSizeArray(i),
	  		          blockSize, otl);
		  baseNetwork->add(module);
	  }
	  else {
		  module = new Linear(hiddenLayerSizeArray(i - 1), hiddenLayerSizeArray(i),
	  				  blockSize, otl);
		  baseNetwork->add(module);
	  }
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
  // Create outputNetwork => softmax layers for tree
  if (name == OVNB) {
  	  outputNetwork = (LinearSoftmax**)(new LinearSoftmax_Bayes*[outputNetworkNumber]);
  	  LinearSoftmax_Bayes* sl = new LinearSoftmax_Bayes(hiddenLayerSize, outputNetworkSize(0),
  			  blockSize, otl);
  	  outputNetwork[0] = sl;
  	  for (i = 1; i < outputNetworkNumber; i++)
	  {
  		  sl = new LinearSoftmax_Bayes(hiddenLayerSize, outputNetworkSize(i), 1, otl);
  		  outputNetwork[i] = sl;
	  }
  }
  else {
  	  outputNetwork = new LinearSoftmax*[outputNetworkNumber];
  	  LinearSoftmax* sl = new LinearSoftmax(hiddenLayerSize, outputNetworkSize(0),
  		  blockSize, otl);
  	  outputNetwork[0] = sl;
  	  for (i = 1; i < outputNetworkNumber; i++)
  	  {
  		  sl = new LinearSoftmax(hiddenLayerSize, outputNetworkSize(i), 1, otl);
  		  outputNetwork[i] = sl;
  	  }
  }
  doneForward.resize(outputNetworkNumber, 1);
  // contextFeature is the last hidden layer
  contextFeature = baseNetwork->output;
  gradContextFeature.resize(contextFeature);
  localCodeWord.resize(blockSize, maxCodeWordLength);
  // Tie word spaces in case of LBL
  if (name == LBL)
    {
      outputNetwork[0]->weight.tieData(baseNetwork->lkt->weight);
    }
  // Create NgramDataSet to manipulate data
  dataSet = new NgramDataSet(ngramType, n, BOS, inputVoc, outputVoc, mapIUnk,
      mapOUnk, BLOCK_NGRAM_NUMBER);
  delete otl;
}
NgramModel::NgramModel(string name, char* inputVocFileString,
    char* outputVocFileString, int mapIUnk, int mapOUnk, int BOS,
    int blockSize, int n, int dimensionSize, string nonLinearType,
    intTensor& hiddenLayerSizeArray, char* codeWordFileString,
    char* outputNetworkSizeFileString)
{
  recurrent = 0;
  this->name = name;
  this->ngramType = 0;
  // Read vocabularies
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
  ioFile readIof;
  // Read a tree structure, codeWord, outputNetworkSize
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

NgramModel::NgramModel(string name, int inputSize, int outputSize,
    int blockSize, int n, int dimensionSize, string nonLinearType,
    intTensor& hiddenLayerSizeArray, intTensor& codeWord,
    intTensor& outputNetworkSize)
{
  recurrent = 0;
  this->name = name;
  this->ngramType = 0;
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
  this->codeWord = codeWord;
  this->outputNetworkSize = outputNetworkSize;

  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  hiddenNumber = hiddenLayerSizeArray.length;
  maxCodeWordLength = this->codeWord.size[1];
  outputNetworkNumber = outputNetworkSize.size[0];
  allocation();
}

void
NgramModel::firstTime()
{
}
void
NgramModel::firstTime(intTensor& context)
{
}

int
NgramModel::train(char* dataFileString, int maxExampleNumber, int iteration,
    string learningRateType, float learningRate, float learningRateDecay)
{
  firstTime();
  ioFile dataIof;
  dataIof.takeReadFile(dataFileString);
  // Read header 2 integers
  int ngramNumber;
  dataIof.readInt(ngramNumber);
  // N is order in data file, can be larger than n, order of model
  int N;
  dataIof.readInt(N);
  if (N < n)
    {
      cerr << "ERROR: N in data is wrong:" << N << " < " << n << endl;
      exit(1);
    }
  // maxExampleNumber = 0 => use all ngrams
  if (maxExampleNumber > ngramNumber || maxExampleNumber == 0)
    {
      maxExampleNumber = ngramNumber;
    }
  float currentLearningRate;
  // nstep is the number of seen examples
  int nstep;
  // nstep at the beginning: number of examples used in
  // previous iterations, here being computed approximately
  nstep = maxExampleNumber * (iteration - 1);
  // In data file, n-grams is written in row major order but tensor is by
  // default in column major order, need to do some tricks here
  // In file 1 2 3 4 5 6 7 8 => 2 4-grams 1 2 3 4, 5 6 7 8
  // readTensor call readStrip to read data from file without header, and
  // in row major order, after reading
  // readTensor.data = [1, 5, 2, 6, 3, 7, 4, 8]
  // readTensor considers data as a matrix (2 x 4):
  // 1 2 3 4
  // 5 6 7 8
  intTensor readTensor(blockSize, N);
  intTensor context;
  intTensor word;
  // context points to context words in readTensor, suppose we use only 3-grams
  // After .sub:
  // 2 3
  // 6 7
  // After .t (each column is for one example in block):
  // 2 6
  // 3 7
  context.sub(readTensor, 0, blockSize - 1, N - n, N - 2);
  context.t();
  // word is the last column of readTensor (as it is actually a vector, we don't
  // need to transpose
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
      // Read a block of n-grams as shown above, context and word already
      // point to a part of data in readTensor
      readTensor.readStrip(&dataIof);
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
  // The last block wit less number of examples than blockSize
  if (remainingNumber != 0 && !dataIof.getEOF())
    {
      context = 0;
      word = SIGN_NOT_WORD;
      // read only remainingNumber examples, copy to readTensor
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
NgramModel::forwardProbability(intTensor& ngramTensor, floatTensor& probTensor)
{
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
  // context points to the context part of ngramTensor
  context.sub(ngramTensor, 0, ngramNumber - 1, 0, n - 2);
  // contextFlag is the last column which represents the index of a first next n-gram
  // which does not have the same context, ngramTensor is supposed to be sorted
  // (all n-grams with the same context are grouped together)
  intTensor contextFlag;
  contextFlag.select(ngramTensor, 1, n + 2);
  // word points to the predicted words of ngramTensor
  intTensor word;
  word.select(ngramTensor, 1, n - 1);
  // Index of ngram in text file, required to write to probTensor
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
      // Read blockSize different contexts in ngramTensor
      while (rBlockSize < blockSize && ngramId < ngramNumber)
        {
          selectBContext.select(bContext, 1, rBlockSize);
          selectContext.select(context, 0, ngramId);
          selectBContext.copy(selectContext);
          ngramId = contextFlag(ngramId);
          rBlockSize++;
        }
      rBlockSize = 0;
      // firstTime does something only with recurrent models
      firstTime(bContext);
      // Forward with baseNetwork and the main softmax layer
      // in bunch mode
      baseNetwork->forward(bContext);
      mainProb = outputNetwork[0]->forward(contextFeature);

      // For each context, for each predicted word, forward softmax layers
      // depending on their codes
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
                      // doneForward is used to guarantee that
                      // for each context, we forward only once for each softmax layer
                      if (!doneForward(idParent))
                        {
                          selectContextFeature.select(contextFeature, 1,
                              rBlockSize);
                          outputNetwork[idParent]->forward(selectContextFeature);
                          doneForward(idParent) = 1;
                        }
                      localProb *= outputNetwork[idParent]->output(localWord);
                    }
                  if (incrUnk != 1)
                    {
                      if (idWord == outputVoc->unk)
                        {
                          localProb = localProb * incrUnk;
                        }
                    }
                  // Sometimes we have 0 probability and we want to write small number
                  // but not zero
                  if (localProb != 0)
                    {
                      probTensor(order(ngramId2)) = localProb;
                    }
                  else
                    {
                      probTensor(order(ngramId2)) = PROBZERO;
                    }
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
  return 1;
}

void
NgramModel::read(ioFile* iof, int allocation, int blockSize)
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
  iof->readInt(maxCodeWordLength);
  iof->readInt(outputNetworkNumber);
  codeWord.resize(outputVoc->wordNumber, maxCodeWordLength);
  outputNetworkSize.resize(outputNetworkNumber, 1);
  codeWord.read(iof);
  outputNetworkSize.read(iof);
  hiddenLayerSizeArray.resize(hiddenNumber, 1);
  hiddenLayerSizeArray.read(iof);
  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  if (allocation)
    {
      this->allocation();
    }
  baseNetwork->read(iof);
  int i;
  if (name != LBL)
    {
      for (i = 0; i < outputNetworkSize.size[0]; i++)
        {
          outputNetwork[i]->read(iof);
        }
    }
  inputVoc->read(iof);
  outputVoc->read(iof);
}

void
NgramModel::write(ioFile* iof)
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
  iof->writeInt(maxCodeWordLength);
  iof->writeInt(outputNetworkNumber);
  codeWord.write(iof);
  outputNetworkSize.write(iof);
  hiddenLayerSizeArray.write(iof);
  baseNetwork->write(iof);
  int i;
  if (name != LBL)
    {
      for (i = 0; i < outputNetworkSize.size[0]; i++)
        {
          outputNetwork[i]->write(iof);
        }
    }
  inputVoc->write(iof);
  outputVoc->write(iof);
  iof->freeWriteFile();
}
