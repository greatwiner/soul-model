#include "mainModel.H"

MultiplesNeuralModel::MultiplesNeuralModel() {

}

MultiplesNeuralModel::MultiplesNeuralModel(int modelNumber) {
	this->modelNumber = modelNumber;
	this->models = new NeuralModel*[modelNumber];
}

MultiplesNeuralModel::~MultiplesNeuralModel() {
	delete[] models;
}

int
MultiplesNeuralModel::decodeWord(int modelIndex, intTensor& word) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return -1;
	}
	return models[modelIndex]->decodeWord(word);
}

int
MultiplesNeuralModel::decodeWord(int modelIndex, intTensor& word, int subBlockSize) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return -1;
	}
	return models[modelIndex]->decodeWord(word, subBlockSize);
}

void
MultiplesNeuralModel::setWeight(int modelIndex, char* layerName, floatTensor& tensor) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return;
	}
	models[modelIndex]->setWeight(layerName, tensor);
}

floatTensor&
MultiplesNeuralModel::getWeight(int modelIndex, char* layerName) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
	}
	return models[modelIndex]->getWeight(layerName);
}

void
MultiplesNeuralModel::setWeightDecay(int modelIndex, float weightDecay) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return;
	}
	models[modelIndex]->setWeightDecay(weightDecay);
}

void
MultiplesNeuralModel::changeBlockSize(int modelIndex, int blockSize) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return;
	}
	models[modelIndex]->changeBlockSize(blockSize);
}

void
MultiplesNeuralModel::changeBlockSize(int blockSize) {
	for (int i = 0; i < modelNumber; i ++) {
		changeBlockSize(i, blockSize);
	}
}

void
MultiplesNeuralModel::trainOne(int modelIndex, intTensor& context, intTensor& word, float learningRate) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return;
	}
	models[modelIndex]->trainOne(context, word, learningRate);
}

void
MultiplesNeuralModel::trainOne(int modelIndex, intTensor& context, intTensor& word, float learningRate, int subBlockSize) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return;
	}
	models[modelIndex]->trainOne(context, word, learningRate, subBlockSize);
}

floatTensor&
MultiplesNeuralModel::forwardOne(int modelIndex, intTensor& context, intTensor& word) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
	}
	return models[modelIndex]->forwardOne(context, word);
}

floatTensor&
MultiplesNeuralModel::computeProbability(int modelIndex, char* textFileName, string textType) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
	}
	return models[modelIndex]->computeProbability(models[modelIndex]->dataSet, textFileName, textType);
}

floatTensor&
MultiplesNeuralModel::computeProbability(int modelIndex) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
	}
	return models[modelIndex]->computeProbability();
}

float
MultiplesNeuralModel::computePerplexity(int modelIndex, char* textFileName, string textType) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return -1;
	}
	return models[modelIndex]->computePerplexity(models[modelIndex]->dataSet, textFileName, textType);
}

float
MultiplesNeuralModel::computePerplexity(int modelIndex) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return -1;
	}
	return models[modelIndex]->computePerplexity();
}

int
MultiplesNeuralModel::forwardProbability(int modelIndex, intTensor& ngramTensor, floatTensor& probTensor) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return -1;
	}
	return models[modelIndex]->forwardProbability(ngramTensor, probTensor);
}

// Only for recurrent models
void
MultiplesNeuralModel::firstTime(int modelIndex) {
	models[modelIndex]->firstTime();
}
void
MultiplesNeuralModel::firstTime(int modelIndex, intTensor& context) {
	models[modelIndex]->firstTime(context);
}
void
MultiplesNeuralModel::firstTime() {
	for (int i = 0; i < modelNumber; i ++) {
		models[i]->firstTime();
	}
}

int
MultiplesNeuralModel::sequenceTrain(char* prefixModel, int gz, char* prefixData,
		int maxExampleNumber, char* trainingFileName, char* validationFileName, string validType, string learningRateType, int minIteration, int maxIteration) {
	int maxExampleNumberArray[modelNumber];
	for (int i = 0; i < modelNumber; i ++) {
		maxExampleNumberArray[i] = maxExampleNumber;
	}
	this->sequenceTrain(prefixModel, gz, prefixData, maxExampleNumberArray, trainingFileName, validationFileName, validType, learningRateType, minIteration, maxIteration);
}

int
MultiplesNeuralModel::sequenceTrain(char* prefixModel, int gz, char* prefixDatas,
	    int* maxExampleNumber, char* trainingFileNames, char* validationFileNames, string validType, string learningRateType, int minIteration, int maxIteration) {
	float learningRateForRd;
	float learningRateForParas;
	float learningRateDecay;
	float weightDecay;
	int computeDevPer = 1;
	float perplexity[modelNumber];
	float prePerplexity[modelNumber];
	char* dataFileName[modelNumber];
	for (int i = 0; i < modelNumber; i ++) {
		dataFileName[i] = new char[260];
	}
	char outputModelFileName[260];
	char convertStr[260];
	char modelIndex[260];
	ioFile iofC;
	int dataC;
	int modelC;
	computeDevPer = iofC.check(validationFileNames, 0);
	time_t start, end;
	int iteration;
	int divide = 0;
	ioFile parasIof;
	floatTensor parasTensor(6, 1);
	parasIof.format = TEXT;
	ioFile modelIof;
	int stop = 0;

	// Read parameters (learningRate, weightDecay, blockSize...) in *.par
	sprintf(convertStr, "%d", minIteration - 1);
	strcpy(outputModelFileName, prefixModel);
	strcat(outputModelFileName, convertStr);
	strcat(outputModelFileName, ".par");
	int paraC = iofC.check(outputModelFileName, 1);
	if (!paraC) {
		return 0;
	}
	parasIof.takeReadFile(outputModelFileName);
	parasTensor.read(&parasIof);
	learningRateForRd = parasTensor(0);
	learningRateForParas = parasTensor(1);
	learningRateDecay = parasTensor(2);
	weightDecay = parasTensor(3);
	changeBlockSize((int) parasTensor(4));
	if (learningRateType == LEARNINGRATE_DOWN) {
		divide = (int) parasTensor(5);
	}
	// Now iteration is the number of first new model

	if (learningRateType == LEARNINGRATE_NORMAL) {
		cout << "Paras (normal): " << learningRateForRd << " " << learningRateForParas << " " << learningRateDecay
	          << " " << weightDecay << " " ;
		for (int i = 0; i < this->modelNumber; i ++) {
			cout << models[i]->blockSize << " ";
		}
		cout << endl;
	}
	else if (learningRateType == LEARNINGRATE_DOWN) {
		cout << "Paras (down): " << learningRateForRd << " " << learningRateForParas << " " << learningRateDecay
	          << " " << weightDecay << " ";
		for (int i = 0; i < this->modelNumber; i ++) {
			cout << models[i]->blockSize << " ";
		}
		cout << divide << endl;
	}

	for (int i = 0; i < this->modelNumber; i ++) {
		setWeightDecay(i, weightDecay);
	}
	// Compute perplexity of dev data, for early stopping
	ioFile validationFiles;
	validationFiles.takeReadFile(validationFileNames);
	char* validationFileName[modelNumber];
	for (int i = 0; i < modelNumber; i++) {
		string filename;
		validationFiles.getLine(filename);
		validationFileName[i] = new char[filename.size() + 1];
		std::copy(filename.begin(), filename.end(), validationFileName[i]);
		validationFileName[i][filename.size()] = '\0';
	}
	if (computeDevPer) {
		time(&start);
		cout << "Compute validation perplexity:" << endl;

		for (int i = 0; i < modelNumber; i++) {
			computePerplexity(i, validationFileName[i], validType);
			prePerplexity[i] = models[i]->dataSet->perplexity;
			cout << "With epoch " << minIteration - 1 << ", perplexity of "
					<< validationFileName[i] << " is " << models[i]->dataSet->perplexity
					<< " ("
					<< models[i]->dataSet->ngramNumber << " ngrams)" << endl;
		}
		time(&end);
		cout << "Finish after " << difftime(end, start) << " seconds"
			 << endl;
	}

	// the name of the files containing perplexities on validation set
	char* outputPerplexityFileNames[modelNumber];
	ofstream* outputPerp = new ofstream[modelNumber];
	for (int i = 0; i < modelNumber; i ++) {
		outputPerplexityFileNames[i] = new char[260];
		strcpy(outputPerplexityFileNames[i], prefixModel);
		strcat(outputPerplexityFileNames[i], "out.per");
		sprintf(modelIndex, "%d", i);
		strcat(outputPerplexityFileNames[i], modelIndex);
		// the file containing perplexities on validation set
		outputPerp[i].open(outputPerplexityFileNames[i], ios_base::app);
	}

	// name of the file containing execution time
	char outputTimeExeFileName[260];
	strcpy(outputTimeExeFileName, prefixModel);
	strcat(outputTimeExeFileName, "out.Time");

	ofstream outputTimeFile;
	outputTimeFile.open(outputTimeExeFileName, ios_base::app);

	// execution time
	float timeExe = 0;

	// Now, train a model
	ioFile prefixDataFile;
	prefixDataFile.takeReadFile(prefixDatas);
	char* prefixData[modelNumber];
	for (int i = 0; i < modelNumber; i++) {
		string filename;
		prefixDataFile.getLine(filename);
		prefixData[i] = new char[filename.size() + 1];
		std::copy(filename.begin(), filename.end(), prefixData[i]);
		prefixData[i][filename.size()] = '\0';
	}
	for (iteration = minIteration; iteration < maxIteration + 1; iteration++) {
		cout << "Iteration: " << iteration << endl;
		sprintf(convertStr, "%d", iteration);
		for (int i = 0; i < modelNumber; i ++) {
			strcpy(dataFileName[i], prefixData[i]);
			strcat(dataFileName[i], convertStr);
			dataC = iofC.check(dataFileName[i], 0);
			if (!dataC) {
				strcat(dataFileName[i], ".gz");
				dataC = iofC.check(dataFileName[i], 0);
				if (!dataC) {
					cout << "Train data file " << convertStr << " does not exist"
						 << endl;
					return 0;
				}
			}
		}
		strcpy(outputModelFileName, prefixModel);
		strcat(outputModelFileName, convertStr);
		if (gz) {
			strcat(outputModelFileName, ".gz");
		}
		modelC = iofC.check(outputModelFileName, 0);
		if (modelC) {
			cerr << "WARNING: Train model file " << convertStr << " exists"
				 << endl;
			return 0;
		}
		time(&start);
		if (learningRateType == LEARNINGRATE_NORMAL) {
			cout << "Paras (normal): " << learningRateForRd << " " << learningRateForParas << " "
				 << learningRateDecay << " " << weightDecay << " ";
			for (int i = 0; i < modelNumber; i ++) {
				cout << models[i]->blockSize;
			}
			cout << " , ";
		}
		else if (learningRateType == LEARNINGRATE_DOWN) {
			cout << "Paras (down): " << learningRateForRd << " " << learningRateForParas << " " << learningRateDecay
				 << " " << weightDecay << " ";
			for (int i = 0; i < modelNumber; i ++) {
				cout << models[i]->blockSize;
			}
			cout << " " << divide
				 << " , ";
			if (divide) {
				learningRateForParas = learningRateForParas / learningRateDecay;
			}
		}
		else if (learningRateType == LEARNINGRATE_ADJUST) {
			cout << "Paras (adjust): " << learningRateForRd << " " << learningRateForParas << " " << learningRateDecay
				 << " " << weightDecay << " ";
			for (int i = 0; i < modelNumber; i ++) {
				cout << models[i]->blockSize;
			}
			cout << " , ";
		}
		int outTrain;
		outTrain = train(dataFileName, maxExampleNumber, iteration,
				  learningRateType, learningRateForParas, learningRateDecay);
		if (outTrain == 0) {
			cerr << "ERROR: Can't finish training" << endl;
			exit(1);
		}
		time(&end);
		cout << "Finish after " << difftime(end, start) / 60 << " minutes"
			 << endl;
		timeExe += difftime(end, start);

		int upDivide = 0;
		if (computeDevPer) {
			// calculate execution time
			time_t start, end;
			time(&start);

			cout << "Compute validation perplexity:" << endl;
			for (int i = 0; i < modelNumber; i ++) {
				forwardProbability(i, models[i]->dataSet->dataTensor, models[i]->dataSet->probTensor);
				prePerplexity[i] = models[i]->dataSet->perplexity;
				perplexity[i] = models[i]->dataSet->computePerplexity();

				cout << "With epoch " << iteration << ", perplexity of "
					 << validationFileName[i] << " is " << models[i]->dataSet->perplexity << " ("
					 << models[i]->dataSet->ngramNumber << " ngrams)" << endl;
			}
			time(&end);

			cout << "Finish after " << difftime(end, start) / 60 << " minutes"
				 << endl;
			int divide_adjust = 0;
			for (int i = 0; i < modelNumber; i ++) {
				if (isnan(perplexity[i])) {
					cout << "Perplexity is NaN" << endl;
					stop = 1;
				}
				else if (perplexity[i] > prePerplexity[i]) {
					cout << "WARNING: Perplexity increases for validation file: " << validationFileName[i] << endl;
					upDivide = 1;
					divide_adjust = 1;
				}
			}
			if (learningRateType == LEARNINGRATE_ADJUST && divide_adjust == 1) {
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
					for (int i = 0; i < modelNumber; i ++) {
						perplexity[i] = prePerplexity[i];
						models[i]->dataSet->perplexity = prePerplexity[i];
					}
				}
				else {
					cout << "We do not back to the precedent model" << endl;
				}
			}
			else {
				if (learningRateType == LEARNINGRATE_ADJUST) {
					learningRateForParas = learningRateForParas*ACC_RATE;
				}
				if (learningRateType == LEARNINGRATE_DOWN) {
					int down_upDivide = 0;
					for (int i = 0; i < modelNumber; i ++) {
						if (log(perplexity[i]) * MUL_LOGLKLHOOD > log(prePerplexity[i])) {
							down_upDivide = 1;
						}
					}
					if (down_upDivide == 1) {
						upDivide = 1;
					}
				}
			}
			// write validation perplexity on file
			for (int i = 0; i < modelNumber; i ++) {
				outputPerp[i] << iteration << " " << perplexity[i] << endl;
			}
		}

		if (strcmp(prefixModel, "xxx")) {
			cout << "NeuralModel::sequenceTrain write here" << endl;
			modelIof.takeWriteFile(outputModelFileName);
			write(&modelIof, 1);
		}

		if (divide == 0 && upDivide == 1) {
			divide = 1;
		}
		else if (divide >= 1) {
			divide++;
		}
		strcat(outputModelFileName, ".par");
		parasTensor(0) = learningRateForRd;
		parasTensor(1) = learningRateForParas;
		parasTensor(2) = learningRateDecay;
		parasTensor(3) = weightDecay;
		parasTensor(4) = models[0]->blockSize;
		if (learningRateType == LEARNINGRATE_DOWN) {
			parasTensor(5) = divide;
		}
		if (strcmp(prefixModel, "xxx")) {
			parasIof.takeWriteFile(outputModelFileName);
			parasTensor.write(&parasIof);
			parasIof.freeWriteFile();
		}

		if (divide >= MAX_DIVIDE && name != OVNB) {
			stop = 1;
		}
		if (stop == 1) {
			break;
		}
		outputTimeFile << iteration << " " << timeExe << endl;
	}
	outputTimeFile.close();

	if (!strcmp(prefixModel, "xxx") && (minIteration != maxIteration)) {
		modelIof.takeWriteFile(outputModelFileName);
		write(&modelIof, 1);
	}
	for (int i = 0; i < modelNumber; i ++) {
		delete[] dataFileName[i];
		delete[] validationFileName[i];
		delete[] prefixData[i];
		outputPerp[i].close();
		delete[] outputPerplexityFileNames[i];
	}
	delete[] outputPerp;
	return 1;
}

int
MultiplesNeuralModel::trainTest(int modelIndex, int maxExampleNumber, float weightDecay, string learningRateType, float learningRate, float learningRateDecay, intTensor& gcontext, intTensor& gword) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return -1;
	}
	return models[modelIndex]->trainTest(maxExampleNumber, weightDecay, learningRateType, learningRate, learningRateDecay, gcontext, gword);
}

int
MultiplesNeuralModel::train(int modelIndex, char* dataFileName, int maxExampleNumber, int iteration, string learningRateType, float learningRate, float learningRateDecay) {
	if (modelIndex < 0 || modelIndex >= this->modelNumber) {
		cout << "MultiplesNeuralModel has only " << this->modelNumber << " models" << endl;
		return -1;
	}
	return models[modelIndex]->train(dataFileName, maxExampleNumber, iteration, learningRateType, learningRate, learningRateDecay);
}
