#include "mainModel.H"

NgramModel_Bayes::NgramModel_Bayes() : NgramModel() {
	this->trainLikel = 0;
}

NgramModel_Bayes::NgramModel_Bayes(string name, int inputSize, int outputSize, int blockSize, int n,
	      int projectionDimension, string nonLinearType,
	      intTensor& hiddenLayerSizeArray, intTensor& codeWord,
	      intTensor& outputNetworkSize) : NgramModel(name, inputSize, outputSize, blockSize, n,
	    	      projectionDimension, nonLinearType,
	    	      hiddenLayerSizeArray, codeWord,
	    	      outputNetworkSize) {
	this->trainLikel = 0;
}

NgramModel_Bayes::NgramModel_Bayes(string name, char* inputVocFileName, char* outputVocFileName,
	      int mapIUnk, int mapOUnk, int BOS, int blockSize, int n,
	      int projectionDimension, string nonLinearType,
	      intTensor& hiddenLayerSizeArray, char* codeWordFileName,
	      char* outputNetworkSizeFileName) : NgramModel(name, inputVocFileName, outputVocFileName,
	    	      mapIUnk, mapOUnk, BOS, blockSize, n,
	    	      projectionDimension, nonLinearType,
	    	      hiddenLayerSizeArray, codeWordFileName,
	    	      outputNetworkSizeFileName) {
	this->trainLikel = 0;
}

NgramModel_Bayes::~NgramModel_Bayes()
{

}

void
NgramModel_Bayes::initializeP() {
	if (Sequential_Bayes* dnmc = static_cast<Sequential_Bayes*>(this->baseNetwork)) {
		dnmc->initializeP();
	}
	for (int i = 0; i < this->outputNetworkNumber; i ++) {
		if (LinearSoftmax_Bayes* sc = static_cast<LinearSoftmax_Bayes*>(this->outputNetwork[i])) {
			sc->initializeP();
		}
	}
}

void
NgramModel_Bayes::firstTime()
{
	// randomly sampling p
	initializeP();
}

void
NgramModel_Bayes::resetGradients() {
	this->baseNetwork->lkt->gradWeight=0;
	for (int i = 0; i < this->baseNetwork->size; i ++) {
		if (Linear_Bayes* d1 = dynamic_cast<Linear_Bayes*>(this->baseNetwork->modules[i])) {
			d1->gradWeight=0;
			d1->gradBias=0;
		}
	}
	for (int i = 0; i < this->outputNetworkNumber; i ++) {
		static_cast<LinearSoftmax_Bayes*>(this->outputNetwork[i])->gradWeight=0;
		static_cast<LinearSoftmax_Bayes*>(this->outputNetwork[i])->gradBias=0;
	}
}

void
NgramModel_Bayes::forwardBackwardAllData(char* dataFileString, int maxExampleNumber, int iteration, int* numberExamples,
		float learningRate, int* accept) {
	// we do forward and backward, and calculate the gradients, so all gradients have to be
	// reset
    this->resetGradients();

    // data file
	ioFile dataIof;
	dataIof.takeReadFile(dataFileString);

	// the number of ngrams read from data file
	int ngramNumber;
	dataIof.readInt(ngramNumber);

	// degree
	int N;
	dataIof.readInt(N);

	if (N < n) {
		cerr << "ERROR: N in data is wrong:" << N << " < " << n << endl;
		exit(1);
	}

	if (maxExampleNumber > ngramNumber || maxExampleNumber == 0) {
		maxExampleNumber = ngramNumber;
	}

	// number of steps
	int nstep;
	nstep = maxExampleNumber * (iteration - 1);

	// read tensor
	intTensor readTensor(blockSize, N);

	// context, which is a part of readTensor. So for modifying the context, we need only to
	// modify readTensor
	intTensor context;

	// word, which is a part of readTensor. So for modifying the word, we need only to modify
	// readTensor
	intTensor word;
	context.sub(readTensor, 0, blockSize - 1, N - n, N - 2);
	context.t();
	word.select(readTensor, 1, N - 1);

	// current example number
	int currentExampleNumber = 0;

	// the number of notices
	int percent = 1;

	// the number of examples between 2 consecutive notices
	float aPercent = maxExampleNumber * CONSTPRINT;

	// the number of examples used
	float iPercent = aPercent * percent;

	// the number of blocks
	int blockNumber = maxExampleNumber / blockSize;

	// retains the number of examples
	*numberExamples = maxExampleNumber;

	// the number of remaining examples
	int remainingNumber = maxExampleNumber - blockSize * blockNumber;

	// counts on block number
	int i;
	for (i = 0; i < blockNumber; i++) {
		// read from data file, so modify context and word
		readTensor.readStrip(&dataIof);
		if (dataIof.getEOF()) {
			break;
		}
		currentExampleNumber += blockSize;
	      /*if (learningRateType == LEARNINGRATE_NORMAL)
	        {
	          currentLearningRate = learningRate / (1 + nstep * learningRateDecay);
	        }
	      else if (learningRateType == LEARNINGRATE_DOWN)
	        {
	          currentLearningRate = learningRate;
	        }*/
		// when readTensor is charged, these new values are also charged into context and word

		// trainOne = forward + backward + calculate gradients for a block. This function also
		// update all parameters in back-propagation
		// indicates if this is the last block
		int last = 0;
		if (remainingNumber == 0 && i == blockNumber - 1) last = 1;
		trainOne(context, word, learningRate, blockSize, last);
		/*if (acceptForOne == 1 && *accept == 0) {
			*accept = 1;
		}*/
		//probabilityOne = this->forwardOne(context, word);
		/*probabilityOne.write();
		cout << endl;*/
		//cout << this->trainPer << endl;
		/*for (int i = 0; i < probabilityOne.size[0]; i ++) {
		  	  this->trainLikel -= log(probabilityOne(i));
	  	  }*/
		nstep += blockSize;
	#if PRINT_DEBUG
	      if (currentExampleNumber > iPercent)
	        {
	          percent++;
	          iPercent = aPercent * percent;
	          //cout << (REAL) currentExampleNumber / maxExampleNumber << " ... "
	            //  << flush;
	        }
	#endif
	    }
	if (remainingNumber != 0 && !dataIof.getEOF()) {
		context = 0;
		word = SIGN_NOT_WORD;

		// tensor for reading from data file
		intTensor lastReadTensor(remainingNumber, N);
		lastReadTensor.readStrip(&dataIof);

		// for modifying the readTensor, then context and word
		intTensor subReadTensor;
		subReadTensor.sub(readTensor, 0, remainingNumber - 1, 0, N - 1);
		subReadTensor.copy(lastReadTensor);
		if (!dataIof.getEOF()) {
	          /*if (learningRateType == LEARNINGRATE_NORMAL)
	            {
	              currentLearningRate = learningRate / (1 + nstep
	                  * learningRateDecay);
	            }
	          else if (learningRateType == LEARNINGRATE_DOWN)
	            {
	              currentLearningRate = learningRate;
	            }*/
	          // trainOne = forward + backward + calculate gradients
			trainOne(context, word, learningRate, remainingNumber, 1);
			/*if (acceptForOne == 1 && *accept == 0) {
				*accept = 1;
			}*/
	          /*probabilityOne = this->forwardOne(context, word);
	          for (int i = 0; i < probabilityOne.size[0]; i ++) {
	        	  this->trainLikel -= log(probabilityOne(i));
	          }*/
		}
	}
	//for test
	/*if (*accept == 0) {
		cout << "ko chap nhan" << endl;
	}
	else {
		cout << "chap nhan" << endl;
	}*/
}

void
NgramModel_Bayes::trainOne(intTensor& context, intTensor& word, float learningRate,
		int subBlockSize, int last) {
  // decode word to localCodeWord for SOUL structure
  decodeWord(word, subBlockSize);

  // forward and backward and calculate gradient
  forwardBackwardOne(context, word, subBlockSize, last);
}

void
NgramModel_Bayes::forwardBackwardOne(intTensor& context, intTensor& word, int subBlockSize, int last) {

	baseNetwork->forward(context);

	gradContextFeature = 0;

	// counts until blocksize
	int rBlockSize;

	// id of parent
	int idParent;

	// local words, taken from localCodeWord
	intTensor localWord;

	// the id of a local word, used along with idParent
	intTensor idLocalWord(1, 1);

	// localword takes the first codeword elements of all words
	localWord.select(localCodeWord, 1, 1);

	// so forward using the principal output network
	static_cast<LinearSoftmax_Bayes*>(outputNetwork[0])->forward(contextFeature);

	// the localWord can be considered as output of the principal output network
	gradInput = static_cast<LinearSoftmax_Bayes*>(outputNetwork[0])->backward(localWord, last);

	// gradient on the output of the base network
	gradContextFeature.axpy(gradInput, 1);

	// takes one line of local word code
	intTensor oneLocalCodeWord;
	// now we iterate for each word in blocksize words
	for (rBlockSize = 0; rBlockSize < subBlockSize; rBlockSize++) {
		// a column of context feature corresponding to rBlockSize
		selectContextFeature.select(contextFeature, 1, rBlockSize);
		// a column of gradContextFeature
		selectGradContextFeature.select(gradContextFeature, 1, rBlockSize);
		oneLocalCodeWord.select(localCodeWord, 0, rBlockSize);
		for (int i = 2; i < maxCodeWordLength; i += 2) {
			if (oneLocalCodeWord(i) == SIGN_NOT_WORD) {
			  break;
			}
			idLocalWord = oneLocalCodeWord(i + 1); // 3
			idParent = oneLocalCodeWord(i); // 2
			// for each outputNetwork in SOUL, we do a forward step and a backward step
			static_cast<LinearSoftmax_Bayes*>(outputNetwork[idParent])->forward(selectContextFeature);
			gradInput = static_cast<LinearSoftmax_Bayes*>(outputNetwork[idParent])->backward(idLocalWord, last);
			selectGradContextFeature.axpy(gradInput, 1);
		}
	}
	static_cast<Sequential_Bayes*>(baseNetwork)->backward(gradContextFeature, last);
}

void
NgramModel_Bayes::forwardProbabilityAllData(char* dataFileString, int maxExampleNumber, int iteration) {
	// reset the trainLikel
	this->trainLikel=0;

	// data file
	ioFile dataIof;
	dataIof.takeReadFile(dataFileString);

	// the number of ngram read from data file
	int ngramNumber;
	dataIof.readInt(ngramNumber);

	// it seems to be the degree
	int N;
	dataIof.readInt(N);
	// for test
	cout << "N: " << N << endl;
	if (N < n) {
		cerr << "ERROR: N in data is wrong:" << N << " < " << n << endl;
		exit(1);
	}

	if (maxExampleNumber > ngramNumber || maxExampleNumber == 0) {
		maxExampleNumber = ngramNumber;
	}

	// the number of n-gram used since the beginning of the training process
	int nstep;
	nstep = maxExampleNumber * (iteration - 1);

	// tensor read from the data file
	intTensor readTensor(blockSize, N);

	// the context tensor
	intTensor context;

	// the output word tensor
	intTensor word;
	context.sub(readTensor, 0, blockSize - 1, N - n, N - 2);
	// we take the last n-1 words between N words
	context.t();
	word.select(readTensor, 1, N - 1);

	// the current number of examples
	int currentExampleNumber = 0;

	// the number of time to print a progress announce
	int percent = 1;

	// the number of examples equivalent to 5\%
	float aPercent = maxExampleNumber * CONSTPRINT;

	// the number of examples used
	float iPercent = aPercent * percent;

	// the number of blocks
	int blockNumber = maxExampleNumber / blockSize;

	// the remaining number of examples after arranging all n-grams to blocks
	int remainingNumber = maxExampleNumber - blockSize * blockNumber;

	// count the number of block
	int i;
	cout << maxExampleNumber << " examples" << endl;
	for (i = 0; i < blockNumber; i++) {
		readTensor.readStrip(&dataIof); // read file n gram for word and context
		if (dataIof.getEOF()) {
			break;
		}
		currentExampleNumber += blockSize;
		probabilityOne = this->forwardOne(context, word);
		// calculate the -likelihood over a block
		for (int i = 0; i < probabilityOne.size[0]; i ++) {
			this->trainLikel -= log(probabilityOne(i));
		}

		nstep += blockSize;
		#if PRINT_DEBUG
			if (currentExampleNumber > iPercent) {
				percent++;
				iPercent = aPercent * percent;
				//cout << (REAL) currentExampleNumber / maxExampleNumber << " ... "
	              //<< flush;
	        }
		#endif
	}
	if (remainingNumber != 0 && !dataIof.getEOF()) {
		// reset
		context = 0;
		word = SIGN_NOT_WORD;

		// the tensor in order to take the remaining block
		intTensor lastReadTensor(remainingNumber, N);

		// read this block
		lastReadTensor.readStrip(&dataIof);

		// tensor inside the readTensor. readTensor affects context and word, so we modify readTensor,
		// via subReadTensor
		intTensor subReadTensor;
		subReadTensor.sub(readTensor, 0, remainingNumber - 1, 0, N - 1);
		subReadTensor.copy(lastReadTensor);
		if (!dataIof.getEOF()) {
			probabilityOne = this->forwardOne(context, word);
			for (int i = 0; i < probabilityOne.size[0]; i ++) {
				this->trainLikel -= log(probabilityOne(i));
			}
		}
	}
}

void
NgramModel_Bayes::updateAllParameters(float learningRate) {
	// for test
	//cout << this->outputNetworkNumber << endl;
	for (int i = 0; i < this->outputNetworkNumber; i ++) {
		static_cast<LinearSoftmax_Bayes*>(this->outputNetwork[i])->updateParameters(learningRate);
	}
	static_cast<Sequential_Bayes*>(this->baseNetwork)->updateParameters(learningRate);
}

void
NgramModel_Bayes::updateAllRandomness(float learningRate) {
	for (int i = 0; i < this->outputNetworkNumber; i ++) {
		static_cast<LinearSoftmax_Bayes*>(this->outputNetwork[i])->updateRandomness(learningRate);
	}
	static_cast<Sequential_Bayes*>(this->baseNetwork)->updateRandomness(learningRate);
}

float
NgramModel_Bayes::calculeH() {
	// Hamiltonian term
	// Modules'calculeH function calculates the part of Hamiltonian term inside the module: squared sum
	// of p and weight decay term
	float h = static_cast<Sequential_Bayes*>(this->baseNetwork)->calculeH();
	for (int i = 0; i < this->outputNetworkNumber; i ++) {
		h+=static_cast<LinearSoftmax_Bayes*>(this->outputNetwork[i])->calculeH();
	}
	// for test
	cout << "Composition of Hamiltonian: " << endl;
	cout << "p: " << h << endl;
	cout << "Likelihood: " << this->trainLikel << endl;
	return h + this->trainLikel;
}

int
NgramModel_Bayes::train(char* dataFileString, int maxExampleNumber, int iteration,
    string learningRateType, float learningRate, float learningRateDecay) {
	// indicates if the new values are accepted
	int accept=0;

	// the number of examples contained in data file
	int numberExamples=0;

	// the previous value of -training likelihood
	float prevTrainLikel;
	// randomly sampling p
	// calculate the first Hamiltonian
	firstTime();

	// forward all the data file to calculate -training likelihood
	forwardProbabilityAllData(dataFileString, maxExampleNumber, iteration);

	float prevH = calculeH();
	// for test
	cout << "Prev H: " << prevH << endl;
	int Tau = 10;
	for (int subIter = 1; subIter <= Tau; subIter++) {
		// for test
		if (subIter % 1 == 0) {
			cout << subIter << endl;
		}
		if (subIter > 1) {
			// p = p - epsilon*gnew/2
			this->updateAllRandomness(learningRate);
			// wnew = wnew + epsilon*p
			this->updateAllParameters(learningRate);
		}
		// do a forward and backward (and calculate the gradients)
		// gnew = gradM(gnew)
		forwardBackwardAllData(dataFileString, maxExampleNumber, iteration, &numberExamples, learningRate, &accept);
		if (subIter > 1) {
			// p = p - epsilon*gnew/2
			this->updateAllRandomness(learningRate);
		}
	}
	// update Hamiltonian
	// Mnew = findM(wnew)
	prevTrainLikel = this->trainLikel;
	forwardProbabilityAllData(dataFileString, maxExampleNumber, iteration);
	float H = calculeH();
	// for test
	cout << "H: " << H << endl;
	float dH = H-prevH;
	// for test
	cout << "Difference in H: " << dH << endl;
	if (dH < 0 || iteration == 1) {
		accept = 1;
		// for test
		cout << "Iteration: " << iteration << endl;
		cout << "Probability of acceptance: " << 1 << endl;
	}
	else {
		// for test
		cout << "Iteration: " << iteration << endl;
		cout << "Probability of acceptance: " << exp(-dH) << endl;
		float U=((float)rand()/(float)RAND_MAX);
		if (U < exp(-dH)) {
			accept = 1;
		}
		else {
			//accept = 0;
			accept = 1;
		}
	}
  // for test
  if (accept == 1) {
	  cout << "We accept" << endl;
  }
  else {
	  cout << "We do not accept" << endl;
  }
  // we accept or not the new parameters
  reUpdateParameters(accept);
  // if not accept, we restore the value of likelihood
  /*if (accept == 0) {
	  this->trainLikel = prevTrainLikel;
  }*/
  /*char allData[260] = "./allData/train_1";
  forwardProbabilityAllData(allData, maxExampleNumber, iteration);*/

#if PRINT_DEBUG
  cout << endl;
#endif
  // for test
  //cout << "numberExamples: " << numberExamples << endl;
  //this->trainPer = this->trainLikel/numberExamples;
  //*ptAccept = accept;
  return 1;
}

void
NgramModel_Bayes::reUpdateParameters(int accept) {
	// for test
	//cout << "vi 1" << endl;
	static_cast<Sequential_Bayes*>(this->baseNetwork)->reUpdateParameters(accept);
	// for test
	//cout << "vi 2" << endl;
	for (int i = 0; i < this->outputNetworkNumber; i ++) {
		static_cast<LinearSoftmax_Bayes*>(this->outputNetwork[i])->reUpdateParameters(accept);
	}
}

int
NgramModel_Bayes::forwardProbability(intTensor& ngramTensor, floatTensor& probTensor)
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
  context.sub(ngramTensor, 0, ngramNumber - 1, 0, n - 2);
  intTensor contextFlag;
  // ngramTensor contains at n+3 column the contextFlag
  contextFlag.select(ngramTensor, 1, n + 2);
  intTensor word;
  word.select(ngramTensor, 1, n - 1);
  intTensor order;
  // n+2 column of ngramTensor is the order
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
          // contextFlag is to mark the data which will be used in chain
          rBlockSize++;
        }
      rBlockSize = 0; // real block size
      baseNetwork->forward(bContext);
      mainProb = outputNetwork[0]->forward(contextFeature);
      // contextFeature is the output of baseNetwork
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
/*
#if PRINT_DEBUG
      if (ngramId > iPercent)
        {
          percent++;
          iPercent = aPercent * percent;
          cout << (REAL) ngramId / ngramNumber << " ... " << flush;
        }
#endif
*/
    }
  while (ngramId < ngramNumber);
#if PRINT_DEBUG
  cout << endl;
#endif
  return 1;
}
