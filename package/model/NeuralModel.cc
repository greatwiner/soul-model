/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Abstract class for neural network language model,
 functions are described in NeuralModel.H
 *******************************************************************/
#include "mainModel.H"

NeuralModel::NeuralModel()
{
  // By default, don't modify p(<unk>)
  incrUnk = 1;
}

NeuralModel::~NeuralModel()
{
}

int
NeuralModel::decodeWord(intTensor& word)
{
  return decodeWord(word, blockSize);
}
int
NeuralModel::decodeWord(intTensor& word, int subBlockSize)
{
  int nBlockSize;
  for (nBlockSize = 0; nBlockSize < subBlockSize; nBlockSize++)
    {
      if (word(nBlockSize) != SIGN_NOT_WORD)
        {
          selectCodeWord.select(codeWord, 0, word(nBlockSize));
          selectLocalCodeWord.select(localCodeWord, 0, nBlockSize);
          selectLocalCodeWord.copy(selectCodeWord);
        }
      else
        {
          selectLocalCodeWord.select(localCodeWord, 0, nBlockSize);
          selectLocalCodeWord = SIGN_NOT_WORD;
        }
    }
  for (nBlockSize = subBlockSize; nBlockSize < blockSize; nBlockSize++)
    {
      selectLocalCodeWord.select(localCodeWord, 0, nBlockSize);
      selectLocalCodeWord = SIGN_NOT_WORD;
    }
  return 1;
}

void
NeuralModel::setWeight(char* layerString, floatTensor& tensor)
{
  if (!strcmp(layerString, "projection"))
    {
      baseNetwork->lkt->weight.copy(tensor);
    }
  else if (!strcmp(layerString, "prediction"))
    {
      outputNetwork[0]->weight.copy(tensor);
    }
}
floatTensor&
NeuralModel::getWeight(char* layerString)
{
  if (!strcmp(layerString, "projection"))
    {
      return baseNetwork->lkt->weight;
    }
  else if (!strcmp(layerString, "prediction"))
    {
      return outputNetwork[0]->weight;
    }
  return baseNetwork->lkt->weight;
}

void
NeuralModel::setWeightDecay(float weightDecay)
{
  // For LBL, weight of lkt and outputNetwork[0] are tied,
  // so we set weightDecay only for outputNetwork[0], unless
  // weights are updated twice with weightDecay
  if (name != LBL)
    {
      baseNetwork->lkt->weightDecay = weightDecay;
    }
  for (int i = 0; i < baseNetwork->size; i++)
    {
      baseNetwork->modules[i]->weightDecay = weightDecay;
    }
  for (int i = 0; i < outputNetworkNumber; i++)
    {
      outputNetwork[i]->weightDecay = weightDecay;
    }
}
void
NeuralModel::changeBlockSize(int blockSize)
{
  if (this->blockSize != blockSize)
    {
      this->blockSize = blockSize;
      baseNetwork->changeBlockSize(blockSize);
      contextFeature = baseNetwork->output;
      gradContextFeature.resize(contextFeature);
      if (!recurrent || !cont)
        {
          dataSet->blockSize = blockSize;
        }
      //Ranking models don't have outputNetwork
      if (outputNetwork != NULL)
        {
          outputNetwork[0]->changeBlockSize(blockSize);
          probabilityOne.resize(blockSize, 1);
          localCodeWord.resize(blockSize, maxCodeWordLength);
        }
    }
}

void
NeuralModel::trainOne(intTensor& context, intTensor& word, float learningRate)
{
  trainOne(context, word, learningRate, blockSize);
}

void
NeuralModel::trainOne(intTensor& context, intTensor& word, float learningRate,
    int subBlockSize)
{
  intTensor localWord;
  intTensor idLocalWord(1, 1);
  int idParent;
  int i;
  // Copy codeWord of predicted words into localCodeWord
  decodeWord(word, subBlockSize);

  // firstTime is required only for recurrent models, see RRLinear
  firstTime(context);

  // Forward from lkt to the last hidden layer
  baseNetwork->forward(context);

  // Initialize the gradient for the last hidden layer
  gradContextFeature = 0;
  int rBlockSize;
  // Forward for the main softmax layer outputNetwork[0]
  // localWord is the indices of top classes of prediced words,
  // the second line of localCodeWord
  localWord.select(localCodeWord, 1, 1);
  outputNetwork[0]->forward(contextFeature);

  // gradInput is the gradient from the main softmax layer
  gradInput = outputNetwork[0]->backward(localWord);

  // gradContextFeature = gradInput
  gradContextFeature.axpy(gradInput, 1);

  // For each predicted word, forward, backward and update other softmax layers
  intTensor oneLocalCodeWord;
  for (rBlockSize = 0; rBlockSize < subBlockSize; rBlockSize++)
    {
      // Select the columns for each example in the block
      selectContextFeature.select(contextFeature, 1, rBlockSize);
      selectGradContextFeature.select(gradContextFeature, 1, rBlockSize);
      oneLocalCodeWord.select(localCodeWord, 0, rBlockSize);
      for (i = 2; i < maxCodeWordLength; i += 2)
        {
          if (oneLocalCodeWord(i) == SIGN_NOT_WORD)
            {
              break;
            }
          idLocalWord = oneLocalCodeWord(i + 1);
          idParent = oneLocalCodeWord(i);
          outputNetwork[idParent]->forward(selectContextFeature);
          gradInput = outputNetwork[idParent]->backward(idLocalWord);
          outputNetwork[idParent]->updateParameters(learningRate);
          selectGradContextFeature.axpy(gradInput, 1);
        }
    }
  // Now gradContextFeature = sum of gradients of outputNetworks
  baseNetwork->backward(gradContextFeature);
  outputNetwork[0]->updateParameters(learningRate);
  baseNetwork->updateParameters(learningRate);
}
int
NeuralModel::trainTest(int maxExampleNumber, float weightDecay,
    string learningRateType, float learningRate, float learningRateDecay,
    intTensor& gcontext, intTensor& gword)
{
  firstTime();
  intTensor context;
  intTensor word;
  context.resize(gcontext);
  context.copy(gcontext);
  word.resize(gword);
  word.copy(gword);
  float currentLearningRate;
  int nstep;
  nstep = 0;
  int currentExampleNumber = 0;
  int subBlockSize = blockSize;
  int percent = 1;
  float aPercent = maxExampleNumber * CONSTPRINT;
  float iPercent = aPercent * percent;

  int blockNumber = maxExampleNumber / blockSize;
  int remainingNumber = maxExampleNumber - blockSize * blockNumber;
  int i;
  setWeightDecay(weightDecay);
  for (i = 0; i < blockNumber; i++)
    {
      //Read one line and then train
      currentExampleNumber += blockSize;
      if (learningRateType == LEARNINGRATE_NORMAL)
        {
          currentLearningRate = learningRate / (1 + nstep * learningRateDecay);
        }
      else if (learningRateType == LEARNINGRATE_DOWN)
        {
          currentLearningRate = learningRate;
        }
      trainOne(context, word, currentLearningRate, subBlockSize);
      nstep += subBlockSize;
      if (currentExampleNumber > iPercent)
        {
          percent++;
          iPercent = aPercent * percent;
          cout << (float) currentExampleNumber / maxExampleNumber << " ... "
              << flush;
        }
    }
  if (remainingNumber != 0)
    {
      if (learningRateType == LEARNINGRATE_NORMAL)
        {
          currentLearningRate = learningRate / (1 + nstep * learningRateDecay);
        }
      else if (learningRateType == LEARNINGRATE_DOWN)
        {
          currentLearningRate = learningRate;
        }
      trainOne(context, word, currentLearningRate, remainingNumber);
    }
  cout << endl;
  return 1;
}

floatTensor&
NeuralModel::forwardOne(intTensor& context, intTensor& word)
{
  int localWord;
  int idParent;
  int i;
  float localProb;

  decodeWord(word);
  firstTime(context);
  baseNetwork->forward(context);
  int idWord;
  mainProb = outputNetwork[0]->forward(contextFeature);
  intTensor oneLocalCodeWord;
  for (idWord = 0; idWord < blockSize; idWord++)
    {
      oneLocalCodeWord.select(localCodeWord, 0, idWord);
      localWord = oneLocalCodeWord(1);
      localProb = mainProb(localWord, idWord);
      for (i = 2; i < maxCodeWordLength; i += 2)
        {
          if (oneLocalCodeWord(i) == SIGN_NOT_WORD)
            {
              break;
            }
          localWord = oneLocalCodeWord(i + 1);
          idParent = oneLocalCodeWord(i);
          selectContextFeature.select(contextFeature, 1, idWord);
          outputNetwork[idParent]->forward(selectContextFeature);
          localProb *= outputNetwork[idParent]->output(localWord);
        }
      probabilityOne(idWord) = localProb;
    }
  return probabilityOne;
}
floatTensor&
NeuralModel::computeProbability(DataSet* dataset, char* textFileName, string textType)
{
  cout << "Read data" << endl;
  ioFile validIof;
  if (textType == "l")
    {
      validIof.format = TEXT;
      validIof.takeReadFile(textFileName);
      dataset->readTextNgram(&validIof);
    }
  else if (textType == "n")
    {
      validIof.format = TEXT;
      validIof.takeReadFile(textFileName);
      dataset->readText(&validIof);
    }
  else if (textType == "id")
    {
      validIof.format = BINARY;
      validIof.takeReadFile(textFileName);
      dataset->readCoBiNgram(&validIof);
    }
  if (dataset->ngramNumber > BLOCK_NGRAM_NUMBER)
    {
      cerr << "ERROR: Not enough memory" << endl;
      exit(1);
    }

  dataset->createTensor();
  cout << "Finish read " << dataset->ngramNumber << " ngrams" << endl;
  cout << "Compute" << endl;
  forwardProbability(dataset->dataTensor, dataset->probTensor);
  return dataset->probTensor;
}

floatTensor&
NeuralModel::computeProbability()
{
  forwardProbability(dataSet->dataTensor, dataSet->probTensor);
  return dataSet->probTensor;

}

float
NeuralModel::computePerplexity(DataSet* dataset, char* textFileName, string textType)
{

  computeProbability(dataset, textFileName, textType);
  dataset->computePerplexity();
  return dataset->perplexity;

}

float
NeuralModel::computePerplexity()
{
  computeProbability();
  dataSet->computePerplexity();
  return dataSet->perplexity;
}

int
NeuralModel::sequenceTrain(char* prefixModel, int gz, char* prefixData,
    int maxExampleNumber, char* trainingFileName, char* validationFileName, string validType,
    string learningRateType, int minIteration, int maxIteration)
{
  float learningRate;
  float learningRateDecay;
  float weightDecay;
  int computeDevPer = 1;
  float perplexity = 0;
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

  // Read parameters (learningRate, weightDecay, blockSize...) in *.par
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
  // Now iteration is the number of first new model

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
  // Compute perplexity of dev data, for early stopping
  if (computeDevPer)
    {
	  	time(&start);
		cout << "Compute training perplexity:" << endl;

		// compute perplexity in charging all the training file
		computePerplexity(this->trainingDataSet, trainingFileName, validType);

		cout << "Compute validation perplexity:" << endl;

		// compute perplexity in charging all the validation file
		computePerplexity(this->dataSet, validationFileName, validType);
		prePerplexity = this->dataSet->perplexity;
		time(&end);

		cout << "With epoch " << minIteration - 1 << ", perplexity of "
		<< trainingFileName << " is " << trainingDataSet->perplexity
		<< " ("
		<< trainingDataSet->ngramNumber << " ngrams)" << endl;
		cout << "With epoch " << minIteration - 1 << ", perplexity of "
		<< validationFileName << " is " << dataSet->perplexity
		<< " ("
		<< dataSet->ngramNumber << " ngrams)" << endl;
		cout << "Finish after " << difftime(end, start) << " seconds"
	  			 << endl;


      /*cout << "Compute validation perplexity:" << endl;
      time(&start);
      perplexity = computePerplexity(validationFileName, validType);
      time(&end);
      prePerplexity = perplexity;
      cout << "With epoch " << minIteration - 1 << ", perplexity of "
          << validationFileName << " is " << perplexity << " ("
          << dataSet->ngramNumber << " ngrams)" << endl;
      cout << "Finish after " << difftime(end, start) / 60 << " minutes"
          << endl;*/
    }

  // Now, train a model
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
		    // calculate execution time
			time_t start, end;
			cout << "Compute training perplexity:" << endl;
			time(&start);

			// current perplexity on training set
			prePerplexity = trainingDataSet->perplexity;
			forwardProbability(trainingDataSet->dataTensor, trainingDataSet->probTensor);
			trainingDataSet->computePerplexity();

			cout << "Compute validation perplexity:" << endl;
			forwardProbability(dataSet->dataTensor, dataSet->probTensor);
			prePerplexity = perplexity;
			perplexity = dataSet->computePerplexity();
			time(&end);

			cout << "With epoch " << iteration << ", perplexity of "
				 << trainingFileName << " is " << trainingDataSet->perplexity << " ("
				 << trainingDataSet->ngramNumber << " ngrams)" << endl;

			cout << "With epoch " << iteration << ", perplexity of "
			   << validationFileName << " is " << dataSet->perplexity << " ("
			   << dataSet->ngramNumber << " ngrams)" << endl;

			cout << "Finish after " << difftime(end, start) / 60 << " minutes"
			   << endl;

			// write training perplexity on file
			//outputTrainingPerp << trainingDataSet->perplexity << endl;

			// write validation perplexity on file
			//outputPerp << dataSet->perplexity << endl;



          /*cout << "Compute validation perplexity:" << endl;
          time_t start, end;
          time(&start);
          prePerplexity = perplexity;
          perplexity = computePerplexity();
          time(&end);

          cout << "With epoch " << iteration << ", perplexity of "
              << validationFileName << " is " << perplexity << " ("
              << dataSet->ngramNumber << " ngrams)" << endl;
          cout << "Finish after " << difftime(end, start) / 60 << " minutes"
              << endl;*/
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
