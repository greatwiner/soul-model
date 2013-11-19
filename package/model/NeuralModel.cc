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
	float learningRateAdaG;
	time_t start, end;
  intTensor localWord;
  intTensor idLocalWord(1, 1);
  int idParent;
  int i;
  // Copy codeWord of predicted words into localCodeWord
  decodeWord(word, subBlockSize);

  // for test
  //cout << "NeuralModel::trainOne here " << endl;

  // firstTime is required only for recurrent models, see RRLinear
  firstTime(context);
  // for test
  //cout << "NeuralModel::trainOne here 1" << endl;

  // scale learning rate
  if (name == OVN_AG) {
	  // for Down-Bloc-Adag
  	  learningRateAdaG = learningRate*sqrt(dynamic_cast<LinearSoftmax_AG*>(outputNetwork[0])->cumulGradWeight);

  	  // for Bloc-Adag
	  //learningRateAdaG = learningRate;
  }

  // Forward from lkt to the last hidden layer
  // for test
  //cout << "NeuralModel::trainOne here 3" << endl;
  baseNetwork->forward(context);

  // Initialize the gradient for the last hidden layer
  gradContextFeature = 0;
  int rBlockSize;
  // Forward for the main softmax layer outputNetwork[0]
  // localWord is the indices of top classes of prediced words,
  // the second line of localCodeWord
  localWord.select(localCodeWord, 1, 1);
  // for test
  //cout << "NeuralModel::trainOne here 4" << endl;
  outputNetwork[0]->forward(contextFeature);

  // for test
  //cout << "NeuralModel::trainOne here 5" << endl;
  // gradInput is the gradient from the main softmax layer
  if (name == OVN_AG) {
	  gradInput = dynamic_cast<LinearSoftmax_AG*>(outputNetwork[0])->backward(localWord);
  }
  else {
	  gradInput = dynamic_cast<LinearSoftmax*>(outputNetwork[0])->backward(localWord);
  }

  // gradContextFeature = gradInput
  gradContextFeature.axpy(gradInput, 1);

  // For each predicted word, forward, backward and update other softmax layers
  intTensor oneLocalCodeWord;
  for (rBlockSize = 0; rBlockSize < subBlockSize; rBlockSize++)
    {
	  // for test
	  //cout << "NeuralModel::trainOne here 6" << endl;
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
          // for test
          //cout << "NeuralModel::trainOne here 7" << endl;
          if (name == OVN_AG) {
          	  gradInput = dynamic_cast<LinearSoftmax_AG*>(outputNetwork[idParent])->backward(idLocalWord);
          }
          else {
          	  gradInput = dynamic_cast<LinearSoftmax*>(outputNetwork[idParent])->backward(idLocalWord);
          }
          if (name == OVN_AG) {
        	  // for test
        	  //cout << "NeuralModel::trainOne update ovn_ag i: " << i << endl;
        	  outputNetwork[idParent]->updateParameters(learningRateAdaG);
          }
          else {
        	  outputNetwork[idParent]->updateParameters(learningRate);
          }
          selectGradContextFeature.axpy(gradInput, 1);
        }
    }
  // for test
  //cout << "NeuralModel::sequenceTrain here 8" << endl;
  // Now gradContextFeature = sum of gradients of outputNetworks
  baseNetwork->backward(gradContextFeature);
  // for test
  //cout << "NeuralModel::sequenceTrain here 9" << endl;
  if (name == OVN_AG) {
	  // for test
	  //cout << "NeuralModel::trainOne update ovn_ag 0" << endl;
	  outputNetwork[0]->updateParameters(learningRateAdaG);
	  // for test
	  //cout << "NeuralModel::sequenceTrain here 11" << endl;
  }
  else {
	  // for test
	  //cout << "NeuralModel::sequenceTrain here 12" << endl;
	  outputNetwork[0]->updateParameters(learningRate);
	  // for test
	  //cout << "NeuralModel::sequenceTrain here 13" << endl;
  }
  if (name == OVN_AG) {
	  // for test
	  //cout << "NeuralModel::sequenceTrain here 13.1" << endl;
	  baseNetwork->updateParameters(learningRateAdaG);
	  // for test
	  //cout << "NeuralModel::sequenceTrain here 13.2" << endl;
  }
  else {
	  baseNetwork->updateParameters(learningRate);
  }
  // for test
  //cout << "NeuralModel::sequenceTrain here 14" << endl;
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
      // for test
      //cout << "NeuralModel::computeProbability here" << endl;
      validIof.takeReadFile(textFileName);
      // for test
      //cout << "NeuralModel::computeProbability here1" << endl;
      dataset->readText(&validIof);
      // for test
      //cout << "NeuralModel::computeProbability here2" << endl;
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

  // for test
  //cout << "NeuralModel::computeProbability here3" << endl;
  dataset->createTensor();
  // for test
  //cout << "NeuralModel::computeProbability here4" << endl;
  cout << "Finish read " << dataset->ngramNumber << " ngrams" << endl;
  cout << "Compute" << endl;
  forwardProbability(dataset->dataTensor, dataset->probTensor);
  // for test
  //cout << "NeuralModel::computeProbability here5" << endl;
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
	// for test
	//cout << "NeuralModel::computePerplexity here" << endl;
  computeProbability(dataset, textFileName, textType);
  // for test
  //cout << "NeuralModel::computePerplexity here1" << endl;
  dataset->computePerplexity();
  // for test
  //cout << "NeuralModel::computePerplexity here2" << endl;
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
	// for test
	//cout << "NeuralModel::sequenceTrain here 0.1" << endl;
  float learningRateForRd;
  float learningRateForParas;
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
  floatTensor parasTensor(6, 1);
  parasIof.format = TEXT;
  ioFile modelIof;
  int stop = 0;
  // for test
  //cout << "NeuralModel::sequenceTrain here 0.2" << endl;

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
  // for test
  //cout << "NeuralModel::sequenceTrain here 0.3" << endl;
  parasTensor.read(&parasIof);
  // for test
  //cout << "NeuralModel::sequenceTrain here 0.4" << endl;
  learningRateForRd = parasTensor(0);
  learningRateForParas = parasTensor(1);
  learningRateDecay = parasTensor(2);
  weightDecay = parasTensor(3);
  changeBlockSize((int) parasTensor(4));
  if (learningRateType == LEARNINGRATE_DOWN)
    {
      divide = (int) parasTensor(5);
    }
  // Now iteration is the number of first new model

  // for test
  //cout << "NeuralModel::sequenceTrain here 0.5" << endl;
  if (learningRateType == LEARNINGRATE_NORMAL)
    {
      cout << "Paras (normal): " << learningRateForRd << " " << learningRateForParas << " " << learningRateDecay
          << " " << weightDecay << " " << blockSize << endl;
    }
  else if (learningRateType == LEARNINGRATE_DOWN)
    {
      cout << "Paras (down): " << learningRateForRd << " " << learningRateForParas << " " << learningRateDecay
          << " " << weightDecay << " " << blockSize << " " << divide << endl;
    }

  setWeightDecay(weightDecay);
  // Compute perplexity of dev data, for early stopping
  if (computeDevPer)
    {
	  	time(&start);
	  	if (strcmp(trainingFileName, "xxx") != 0) {
	  		cout << "Compute training perplexity:" << endl;

	  		// compute perplexity in charging all the training file
	  		computePerplexity(this->trainingDataSet, trainingFileName, validType);
	  		cout << "With epoch " << minIteration - 1 << ", perplexity of "
	  				<< trainingFileName << " is " << trainingDataSet->perplexity
	  				<< " ("
	  				<< trainingDataSet->ngramNumber << " ngrams)" << endl;
	  	}

		cout << "Compute validation perplexity:" << endl;

		// compute perplexity in charging all the validation file
		// for test
		//cout << "NeuralModel::sequenceTrain here" << endl;
		computePerplexity(this->dataSet, validationFileName, validType);
		// for test
		//cout << "NeuralModel::sequenceTrain here1" << endl;
		prePerplexity = this->dataSet->perplexity;
		// for test
		//cout << "NeuralModel::sequenceTrain here2" << endl;
		time(&end);

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

    // the name of the file containing perplexities on validation set
  	char outputPerplexity[260];
  	strcpy(outputPerplexity, prefixModel);
  	strcat(outputPerplexity, "out.per");

  	// the file containing perplexities on validation set
  	ofstream outputPerp;
  	outputPerp.open(outputPerplexity, ios_base::app);

  	// the name of the file containing perplexities on training set
  	char outputTrainingPerplexity[260];
  	strcpy(outputTrainingPerplexity, prefixModel);
  	strcat(outputTrainingPerplexity, "out.trainPer");

  	// the file containing perplexities on training set
  	ofstream outputTrainingPerp;
  	outputTrainingPerp.open(outputTrainingPerplexity, ios_base::app);

  	// name of the file containing execution time
  	char outputTimeExeFileName[260];
  	strcpy(outputTimeExeFileName, prefixModel);
  	strcat(outputTimeExeFileName, "out.Time");

  	ofstream outputTimeFile;
  	outputTimeFile.open(outputTimeExeFileName, ios_base::app);

  	// execution time
  	float timeExe = 0;

  // Now, train a model
  for (iteration = minIteration; iteration < maxIteration + 1; iteration++)
    {
      cout << "Iteration: " << iteration << endl;
      // for test
      if (name == OVN_AG) {
    	  cout << "NeuralModel::sequenceTrain lkt cumul: " << sqrt(dynamic_cast<LookupTable_AG*>(this->baseNetwork->lkt)->cumulGradWeight.averageSquareBig()) << endl;
    	  //dynamic_cast<LookupTable_AG*>(this->baseNetwork->lkt)->cumulGradWeight = INIT_VALUE_ADAG;
    	  cout << "NeuralModel::sequenceTrain linear cumulWeight: " << dynamic_cast<Linear_AG*>(this->baseNetwork->modules[0])->cumulGradWeight << endl;
    	  //dynamic_cast<Linear_AG*>(this->baseNetwork->modules[0])->cumulGradWeight = INIT_VALUE_ADAG;
    	  cout << "NeuralModel::sequenceTrain linear cumulBias: " << dynamic_cast<Linear_AG*>(this->baseNetwork->modules[0])->cumulGradBias << endl;
    	  //dynamic_cast<Linear_AG*>(this->baseNetwork->modules[0])->cumulGradBias = INIT_VALUE_ADAG;
		  cout << "NeuralModel::sequenceTrain linearsoftmax cumulWeight: " << dynamic_cast<LinearSoftmax_AG*>(this->outputNetwork[0])->cumulGradWeight << endl;
		  //dynamic_cast<LinearSoftmax_AG*>(this->outputNetwork[0])->cumulGradWeight = INIT_VALUE_ADAG;
		  cout << "NeuralModel::sequenceTrain linearsoftmax cumulBias: " << dynamic_cast<LinearSoftmax_AG*>(this->outputNetwork[0])->cumulGradBias << endl;
		  //dynamic_cast<LinearSoftmax_AG*>(this->outputNetwork[0])->cumulGradBias = INIT_VALUE_ADAG;
      }
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
          cout << "Paras (normal): " << learningRateForRd << " " << learningRateForParas << " "
              << learningRateDecay << " " << weightDecay << " " << blockSize
              << " , ";
        }
      else if (learningRateType == LEARNINGRATE_DOWN)
        {
          cout << "Paras (down): " << learningRateForRd << " " << learningRateForParas << " " << learningRateDecay
              << " " << weightDecay << " " << blockSize << " " << divide
              << " , ";
          if (divide)
            {
              learningRateForParas = learningRateForParas / learningRateDecay;
            }
        }
      else if (learningRateType == LEARNINGRATE_ADJUST) {
    	  cout << "Paras (adjust): " << learningRateForRd << " " << learningRateForParas << " " << learningRateDecay
				<< " " << weightDecay << " " << blockSize
				<< " , ";
      }
      int outTrain;
      /*if (this->name == OVNB) {
    	  outTrain = static_cast<NgramModel_Bayes*>(this)->train(dataFileName, maxExampleNumber, iteration,
			  learningRateType, learningRateForRd, learningRateForParas, learningRateDecay);
      }*/
      //else {
      // for test
      //cout << "NeuralModel::sequenceTrain here 2.0" << endl;
	  outTrain = train(dataFileName, maxExampleNumber, iteration,
			  learningRateType, learningRateForParas, learningRateDecay);
      //}
	  // for test
	  //cout << "NeuralModel::sequenceTrain here 2.1" << endl;
      if (outTrain == 0)
        {
          cerr << "ERROR: Can't finish training" << endl;
          exit(1);
        }
      time(&end);
      cout << "Finish after " << difftime(end, start) / 60 << " minutes"
          << endl;
      // accumulate
      timeExe += difftime(end, start);

      int upDivide = 0;
      if (computeDevPer)
        {
		    // calculate execution time
			time_t start, end;
			time(&start);

			if (strcmp(trainingFileName, "xxx") != 0) {
				cout << "Compute training perplexity:" << endl;

				forwardProbability(trainingDataSet->dataTensor, trainingDataSet->probTensor);
				trainingDataSet->computePerplexity();
				cout << "With epoch " << iteration << ", perplexity of "
								 << trainingFileName << " is " << trainingDataSet->perplexity << " ("
								 << trainingDataSet->ngramNumber << " ngrams)" << endl;
				// write training perplexity on file
				outputTrainingPerp << iteration << " " << trainingDataSet->perplexity << endl;
			}

			cout << "Compute validation perplexity:" << endl;
			forwardProbability(dataSet->dataTensor, dataSet->probTensor);
			prePerplexity = dataSet->perplexity;
			perplexity = dataSet->computePerplexity();

			cout << "With epoch " << iteration << ", perplexity of "
			   << validationFileName << " is " << dataSet->perplexity << " ("
			   << dataSet->ngramNumber << " ngrams)" << endl;

			time(&end);

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
              if (learningRateType == LEARNINGRATE_ADJUST) {
            	  learningRateForParas = learningRateForParas / learningRateDecay;
            	  if (UNDO == 1) {
            		  cout << "Back to the precedent model" << endl;
            		  char convertStrPre[260];
            		  sprintf(convertStrPre, "%d", iteration-1);
            		  char modelFileNamePre[260];
            		  strcpy(modelFileNamePre, prefixModel);
            		  strcat(modelFileNamePre, convertStrPre);
            		  modelC = iofC.check(modelFileNamePre, 0);
            		  if (!modelC) {
            			  cerr << "WARNING: Train model file " << modelFileNamePre << " does not exists" << endl;
            			  return 0;
            		  }
            		  modelIof.takeReadFile(modelFileNamePre);
            		  read(&modelIof, 0, (int) parasTensor(4));

            		  // return perplexity to the precedent value
            		  perplexity = prePerplexity;
            		  dataSet->perplexity = prePerplexity;
            	  }
            	  else {
            		  cout << "We do not back to the precedent model" << endl;
            	  }
              }
            }
          else
            {
        	  if (learningRateType == LEARNINGRATE_ADJUST) {
     			  learningRateForParas = learningRateForParas*ACC_RATE;
        	  }
              if (learningRateType == LEARNINGRATE_DOWN)
                {
                  if (log(perplexity) * MUL_LOGLKLHOOD > log(prePerplexity))
                    {
                      upDivide = 1;
                    }
                }
            }
          // write validation perplexity on file
			outputPerp << iteration << " " << perplexity << endl;
        }

      if (strcmp(prefixModel, "xxx")) {
		  //cout << "NeuralModel::sequenceTrain write here" << endl;
		  modelIof.takeWriteFile(outputModelFileName);
		  write(&modelIof, 1);
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
      parasTensor(0) = learningRateForRd;
      parasTensor(1) = learningRateForParas;
      parasTensor(2) = learningRateDecay;
      parasTensor(3) = weightDecay;
      parasTensor(4) = blockSize;
      if (learningRateType == LEARNINGRATE_DOWN)
        {
          parasTensor(5) = divide;
        }
      if (strcmp(prefixModel, "xxx"))
        {
          parasIof.takeWriteFile(outputModelFileName);
          parasTensor.write(&parasIof);
          parasIof.freeWriteFile();
        }

      if (divide >= MAX_DIVIDE && name != OVNB)
        {
          stop = 1;
        }
      if (stop == 1)
        {
          break;
        }
      outputTimeFile << iteration << " " << timeExe << endl;
    }
    outputTimeFile.close();

  	outputPerp.close();

  	outputTrainingPerp.close();

  if (!strcmp(prefixModel, "xxx") && !isnan(perplexity) && (minIteration
      != maxIteration))
    {
      modelIof.takeWriteFile(outputModelFileName);
      write(&modelIof, 1);
    }
  return 1;
}
