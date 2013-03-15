/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for function neural network models
 Input is a vector of float features, output is a class
 *******************************************************************/
#include "mainModel.H"

FunctionModel::FunctionModel()
{
}

FunctionModel::~FunctionModel()
{
  delete baseNetwork;
  delete outputNetwork;
  delete dataSet;
}

void
FunctionModel::allocation()
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
  baseNetwork
      = new FunctionSequential(hiddenLayerSizeArray.length * hiddenStep);
  int i;
  Module* module;
  module = new Linear(dim, hiddenLayerSizeArray(0), blockSize, otl);
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
  outputNetwork = new LinearSoftmax(hiddenLayerSizeArray(i - 1), classNumber,
      blockSize, otl);
  contextFeature = baseNetwork->output;
  dataSet = new FunctionDataSet(dim, classNumber);
  data.resize(dim, blockSize);
  index.resize(blockSize, 1);
  delete otl;
}
FunctionModel::FunctionModel(int dim, int classNumber, int blockSize,
    string nonLinearType, intTensor& hiddenLayerSizeArray)
{
  this->dim = dim;
  this->classNumber = classNumber;
  this->blockSize = blockSize;
  this->nonLinearType = nonLinearType;
  this->hiddenLayerSizeArray.resize(hiddenLayerSizeArray);
  this->hiddenLayerSizeArray.copy(hiddenLayerSizeArray);
  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  hiddenNumber = hiddenLayerSizeArray.length;
  allocation();

}

void
FunctionModel::trainOne(floatTensor& readData, float learningRate,
    int subBlockSize)
{
  data = 0;
  index = -1;
  int x;
  int y;
  for (x = 0; x < subBlockSize; x++)
    {
      for (y = 0; y < dim; y++)
        {
          data(y, x) = readData(x, y); //Attention, readData is transposed
        }
      index(x) = (int) readData(x, dim);
    }
  baseNetwork->forward(data);
  //Special because we forward for the main outputNetwork[0]
  outputNetwork->forward(contextFeature);
  gradInput = outputNetwork->backward(index);
  baseNetwork->backward(gradInput);
  outputNetwork->updateParameters(learningRate);
  baseNetwork->updateParameters(learningRate);
}

int
FunctionModel::train(char* dataFileName, int maxExampleNumber, int iteration,
    string learningRateType, float learningRate, float learningRateDecay)
{
  ioFile dataIof;
  dataIof.takeReadFile(dataFileName);
  int dataNumber;
  dataIof.readInt(dataNumber);
  int readDima1;
  dataIof.readInt(readDima1);
  if (readDima1 != dim + 1)
    {
      cerr << "dim in data is wrong" << endl;
      return 0;
    }

  if (maxExampleNumber > dataNumber || maxExampleNumber == 0)
    {
      maxExampleNumber = dataNumber;
    }

  float currentLearningRate;
  int nstep;
  nstep = maxExampleNumber * (iteration - 1);
  floatTensor readTensor(blockSize, dim + 1);
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
      trainOne(readTensor, currentLearningRate, blockSize);
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
      floatTensor lastReadTensor(remainingNumber, dim + 1);
      lastReadTensor.readStrip(&dataIof);
      floatTensor subReadTensor;
      subReadTensor.sub(readTensor, 0, remainingNumber - 1, 0, dim);
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
          trainOne(readTensor, currentLearningRate, remainingNumber);
        }
    }
#if PRINT_DEBUG
  cout << endl;
#endif
  return 1;
}

void
FunctionModel::setWeightDecay(float weightDecay)
{
  for (int i = 0; i < baseNetwork->size; i++)
    {
      baseNetwork->modules[i]->weightDecay = weightDecay;
    }
  outputNetwork->weightDecay = weightDecay;
}

void
FunctionModel::changeBlockSize(int blockSize)
{
  if (this->blockSize != blockSize)
    {
      this->blockSize = blockSize;
      baseNetwork->changeBlockSize(blockSize);
      contextFeature = baseNetwork->output;
      outputNetwork->changeBlockSize(blockSize);
      data.resize(dim, blockSize);
      index.resize(blockSize, 1);

    }
}
int
FunctionModel::sequenceTrain(char* prefixModel, int gz, char* prefixData,
    int maxExampleNumber, char* validationFileName, string learningRateType,
    int minIteration, int maxIteration)
{
  float learningRate;
  float learningRateDecay;
  float weightDecay;
  int computeDevPer = 1;
  float perplexity;
  float prePerplexity;
  char dataFileName[260];
  char outputModelFileName[260];
  char convertStr[260];
  ioFile iofC;
  int dataC;
  int modelC;
  computeDevPer = iofC.check(validationFileName, 0);
  time_t start, end;
  int iteration;
  int divide = 0;
  ioFile parasIof;
  floatTensor parasTensor;
  parasIof.format = TEXT;
  ioFile modelIof;
  int stop = 0;

  // Read parameters
  sprintf(convertStr, "%d", minIteration - 1);
  strcpy(outputModelFileName, prefixModel);
  strcat(outputModelFileName, convertStr);
  strcat(outputModelFileName, ".par");
  int paraC = iofC.check(outputModelFileName, 1);
  if (!paraC)
    {
      return 0;
    }
  parasIof.takeReadFile(outputModelFileName);
  parasTensor.read(&parasIof);
  learningRate = parasTensor(0);
  learningRateDecay = parasTensor(1);
  weightDecay = parasTensor(2);
  changeBlockSize((int) parasTensor(3));
  if (learningRateType == LEARNINGRATE_DOWN)
    {
      divide = (int) parasTensor(4);
    }
  // Now iteration is number of first new model
  // back to last model

  if (learningRateType == LEARNINGRATE_NORMAL)
    {
      cout << "Paras (normal): " << learningRate << " " << learningRateDecay
          << " " << weightDecay << " " << blockSize << endl;
    }
  else if (learningRateType == LEARNINGRATE_DOWN)
    {
      cout << "Paras (down): " << learningRate << " " << learningRateDecay
          << " " << weightDecay << " " << blockSize << " " << divide << endl;
    }

  setWeightDecay(weightDecay);
  if (computeDevPer)
    {
      cout << "Compute validation perplexity:" << endl;
      time(&start);
      computeForward(validationFileName, 0);
      perplexity = dataSet->computePerplexity();
      time(&end);
      prePerplexity = perplexity;
      cout << "With epoch " << minIteration - 1 << ", perplexity of "
          << validationFileName << " is " << perplexity << " ("
          << dataSet->dataNumber << " ngrams)" << endl;
      cout << "Finish after " << difftime(end, start) / 60 << " minutes"
          << endl;
    }

  for (iteration = minIteration; iteration < maxIteration + 1; iteration++)
    {
      cout << "Iteration: " << iteration << endl;
      sprintf(convertStr, "%d", iteration);
      strcpy(dataFileName, prefixData);
      strcat(dataFileName, convertStr);
      dataC = iofC.check(dataFileName, 0);
      if (!dataC)
        {
          strcat(dataFileName, ".gz");
          dataC = iofC.check(dataFileName, 0);
          if (!dataC)
            {
              cout << "Train data file " << convertStr << " does not exist"
                  << endl;
              return 0;
            }
        }
      strcpy(outputModelFileName, prefixModel);
      strcat(outputModelFileName, convertStr);
      if (gz)
        {
          strcat(outputModelFileName, ".gz");
        }
      modelC = iofC.check(outputModelFileName, 0);
      if (modelC)
        {
          cerr << "WARNING: Train model file " << convertStr << " exists"
              << endl;
          return 0;
        }
      time(&start);
      if (learningRateType == LEARNINGRATE_NORMAL)
        {
          cout << "Paras (normal): " << learningRate << " "
              << learningRateDecay << " " << weightDecay << " " << blockSize
              << " , ";
        }
      else if (learningRateType == LEARNINGRATE_DOWN)
        {
          cout << "Paras (down): " << learningRate << " " << learningRateDecay
              << " " << weightDecay << " " << blockSize << " " << divide
              << " , ";
          if (divide)
            {
              learningRate = learningRate / learningRateDecay;
            }
        }
      int outTrain;
      outTrain = train(dataFileName, maxExampleNumber, iteration,
          learningRateType, learningRate, learningRateDecay);
      if (outTrain == 0)
        {
          cerr << "ERROR: Can't finish training" << endl;
          exit(1);
        }
      time(&end);
      cout << "Finish after " << difftime(end, start) / 60 << " minutes"
          << endl;

      if (strcmp(prefixModel, "xxx"))
        {
          modelIof.takeWriteFile(outputModelFileName);
          write(&modelIof);
        }

      int upDivide = 0;
      if (computeDevPer)
        {
          cout << "Compute validation perplexity:" << endl;
          time_t start, end;
          time(&start);
          prePerplexity = perplexity;
          forward(dataSet->dataTensor, dataSet->probTensor);
          perplexity = dataSet->computePerplexity();
          time(&end);

          cout << "With epoch " << iteration << ", perplexity of "
              << validationFileName << " is " << perplexity << " ("
              << dataSet->dataNumber << " ngrams)" << endl;
          cout << "Finish after " << difftime(end, start) / 60 << " minutes"
              << endl;
          if (isnan(perplexity))
            {
              cout << "Perplexity is NaN" << endl;
              stop = 1;
            }
          else if (perplexity > prePerplexity)
            {
              cout << "WARNING: Perplexity increases" << endl;
              upDivide = 1;
            }
          else
            {
              if (learningRateType == LEARNINGRATE_DOWN)
                {
                  if (log(perplexity) * MUL_LOGLKLHOOD > log(prePerplexity))
                    {
                      upDivide = 1;
                    }
                }
            }
        }
      if (divide == 0 && upDivide == 1)
        {
          divide = 1;
        }
      else if (divide >= 1)
        {
          divide++;
        }
      strcat(outputModelFileName, ".par");
      parasTensor(0) = learningRate;
      parasTensor(1) = learningRateDecay;
      parasTensor(2) = weightDecay;
      parasTensor(3) = blockSize;
      if (learningRateType == LEARNINGRATE_DOWN)
        {
          parasTensor(4) = divide;
        }
      if (strcmp(prefixModel, "xxx"))
        {
          parasIof.takeWriteFile(outputModelFileName);
          parasTensor.write(&parasIof);
          parasIof.freeWriteFile();
        }

      if (divide >= MAX_DIVIDE)
        {
          stop = 1;
        }
      if (stop == 1)
        {
          break;
        }
    }
  if (!strcmp(prefixModel, "xxx") && !isnan(perplexity) && (minIteration
      != maxIteration))
    {
      modelIof.takeWriteFile(outputModelFileName);
      write(&modelIof);
    }
  return 1;
}

int
FunctionModel::forward(floatTensor& dataTensor, floatTensor& probTensor)
{
#if PRINT_CONTEXT
  ioFile iofContext;
  iofContext.binary = 0;
  iofContext.takeWriteFile((char*) "context");
#endif
  int dataNumber = dataTensor.size[0];
  int rBlockSize;
  int oBlockSize;
  int percent = 1;
  float aPercent = dataNumber * CONSTPRINT;
  float iPercent = aPercent * percent;
  int maxBlockSize;
  rBlockSize = 0;
  int dataId;
  int x, y;
  do
    {
      dataId = (rBlockSize + 1) * blockSize;
      maxBlockSize = blockSize;
      if (dataId > dataNumber)
        {
          maxBlockSize = dataNumber - rBlockSize * blockSize;
          dataId = dataNumber;
        }

      for (x = rBlockSize * blockSize; x < dataId; x++)
        {
          for (y = 0; y < dim; y++)
            {
              data(y, x - rBlockSize * blockSize) = dataTensor(x, y);
            }
          index(x - rBlockSize * blockSize) = (int) dataTensor(x, dim);
        }

      baseNetwork->forward(data);
      outputNetwork->forward(contextFeature);

      for (oBlockSize = 0; oBlockSize < maxBlockSize; oBlockSize++)
        {
          probTensor(rBlockSize * blockSize + oBlockSize)
              = outputNetwork->output(index(oBlockSize), oBlockSize);
#if PRINT_CONTEXT
          for (wi = 0; wi < contextFeature.size[0]; wi++)
            {
              *(iofContext.fo) << contextFeature(wi, oBlockSize) << " ";
            }
          *(iofContext.fo) << endl;
#endif
        }
      rBlockSize++;
#if PRINT_DEBUG
      if (dataId > iPercent)
        {
          percent++;
          iPercent = aPercent * percent;
          cout << (float) dataId / dataNumber << " ... " << flush;
        }
#endif
    }
  while (dataId < dataNumber);

#if PRINT_DEBUG
  cout << endl;
#endif
  return 1;
}

int
FunctionModel::computeForward(char* textFileName, int type)
{
  cout << "Read data" << endl;
  ioFile validIof;
  validIof.takeReadFile(textFileName);
  if (type == 0)
    {
      dataSet->readBiNgram(&validIof);
    }
  else if (type == 1)
    {
      dataSet->readAllClassBiNgram(&validIof);
    }
  dataSet->createTensor();
  cout << "Finish read " << dataSet->dataNumber << " ngrams" << endl;
  cout << "Computing" << endl;
  forward(dataSet->dataTensor, dataSet->probTensor);
  return 1;
}

void
FunctionModel::read(ioFile* iof, int allocation, int blockSize)
{
  iof->readString(name);
  iof->readInt(dim);
  iof->readInt(classNumber);
  if (blockSize != 0)
    {
      this->blockSize = blockSize;
    }
  else
    {
      this->blockSize = DEFAULT_BLOCK_SIZE;
    }

  iof->readString(nonLinearType);

  iof->readInt(hiddenNumber);
  hiddenLayerSizeArray.resize(hiddenNumber, 1);
  hiddenLayerSizeArray.read(iof);
  hiddenLayerSize = hiddenLayerSizeArray(hiddenLayerSizeArray.length - 1);
  if (allocation)
    {
      this->allocation();
    }
  baseNetwork->read(iof);
  outputNetwork->read(iof);
}

void
FunctionModel::write(ioFile* iof)
{
  iof->writeString(name);
  iof->writeInt(dim);
  iof->writeInt(classNumber);
  iof->writeString(nonLinearType);
  iof->writeInt(hiddenNumber);
  hiddenLayerSizeArray.write(iof);
  baseNetwork->write(iof);
  outputNetwork->write(iof);
}

