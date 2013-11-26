#include "mainModel.H"

JointNgramWordTranslationModel::JointNgramWordTranslationModel() {

}

JointNgramWordTranslationModel::~JointNgramWordTranslationModel() {
	// for test
	cout << "JointNgramWordTranslationModel::~JointNgramWordTranslationModel here" << endl;
	for (int i = 0; i < modelNumber; i ++) {
		// for test
		cout << "JointNgramWordTranslationModel::~JointNgramWordTranslationModel i: " << i << endl;
		delete this->models[i];
		// for test
		cout << "JointNgramWordTranslationModel::~JointNgramWordTranslationModel here 1" << endl;
	}
	// for test
	cout << "JointNgramWordTranslationModel::~JointNgramWordTranslationModel here 2" << endl;
}

void
JointNgramWordTranslationModel::allocation() {
	models = new NeuralModel*[modelNumber];
	for (int i = 0; i < modelNumber; i ++) {
		// for test
		//cout << "JointNgramWordTranslationModel::allocation i: " << i << endl;
		models[i] = new NgramWordTranslationModel();
	}

	// this outils will be shared by all models, (I suppose)
	otl = new outils();
	otl->sgenrand(time(NULL));
	// for test
	//cout << "JointNgramWordTranslationModel::allocation here 2" << endl;
}

// create a joint model from separate models
// whereInitCommonWeight: we take
JointNgramWordTranslationModel::JointNgramWordTranslationModel(NgramWordTranslationModel** transModels, int modelNumber, int whereInitCommonWeight) {
	this->name = JWTOVN;
	this->modelNumber = modelNumber;
	models = (NeuralModel**)transModels;
	// for test
	//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel1 reference: " << transModels[0]->outputVoc->table[0]->next->next << " index: " << transModels[0]->outputVoc->table[0]->next->next->index << " word: " << transModels[0]->outputVoc->table[0]->next->next->word << endl;
	// for test
	//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel2 reference: " << models[0]->outputVoc->table[0]->next->next << " index: " << models[0]->outputVoc->table[0]->next->next->index << " word: " << models[0]->outputVoc->table[0]->next->next->word << endl;
	// this outils will be shared by all models, (I suppose)
	otl = new outils();
	otl->sgenrand(time(NULL));
	if (whereInitCommonWeight >= 0 && whereInitCommonWeight < modelNumber) {
		// create a new floatTensor using weights from whereInitCommonWeight
		this->jointWeightLkt.copy(this->models[whereInitCommonWeight]->baseNetwork->lkt->weight);
		// for test
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel3" << endl;
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel3 reference: " << models[0]->outputVoc->table[0]->next->next << " index: " << models[0]->outputVoc->table[0]->next->next->index << " word: " << models[0]->outputVoc->table[0]->next->next->word << endl;
	}
	else { // we initialize the weight of the first model, then use it as shared weight
		// for test
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel4" << endl;
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel4 reference: " << models[0]->outputVoc->table[0]->next->next << " index: " << models[0]->outputVoc->table[0]->next->next->index << " word: " << models[0]->outputVoc->table[0]->next->next->word << endl;
		this->models[0]->baseNetwork->lkt->init1class();
		this->jointWeightLkt.copy(this->models[0]->baseNetwork->lkt->weight);
	}
	// for test
	//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel5" << endl;
	//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel5 reference: " << models[0]->outputVoc->table[0]->next->next << " index: " << models[0]->outputVoc->table[0]->next->next->index << " word: " << models[0]->outputVoc->table[0]->next->next->word << endl;

	// very important, shared weights
	for (int i = 0; i < modelNumber; i ++) {
		// for test
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel5.1" << endl;
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel5.1 reference: " << models[0]->outputVoc->table[0]->next->next << " index: " << models[0]->outputVoc->table[0]->next->next->index << " word: " << models[0]->outputVoc->table[0]->next->next->word << endl;
		models[i]->baseNetwork->lkt->shareWeight(jointWeightLkt);
		// for test
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel5.2" << endl;
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel5.2 reference: " << models[0]->outputVoc->table[0]->next->next << " index: " << models[0]->outputVoc->table[0]->next->next->index << " word: " << models[0]->outputVoc->table[0]->next->next->word << endl;
		// for test
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel5.3" << endl;
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel5.3 reference: " << models[0]->outputVoc->table[0]->next->next << " index: " << models[0]->outputVoc->table[0]->next->next->index << " word: " << models[0]->outputVoc->table[0]->next->next->word << endl;
		models[i]->otl = this->otl;
		// for test
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel5.4" << endl;
		//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel5.4 reference: " << models[0]->outputVoc->table[0]->next->next << " index: " << models[0]->outputVoc->table[0]->next->next->index << " word: " << models[0]->outputVoc->table[0]->next->next->word << endl;
	}
	// for test
	//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel6" << endl;
	//cout << "JointNgramWordTranslationModel::JointNgramWordTranslationModel6 reference: " << models[0]->outputVoc->table[0]->next->next << " index: " << models[0]->outputVoc->table[0]->next->next->index << " word: " << models[0]->outputVoc->table[0]->next->next->word << endl;
}

// Read, write with file
void
JointNgramWordTranslationModel::read(ioFile* iof, int allocation, int blockSize) {
	// for test
	//cout << "JointNgramWordTranslationModel::read here" << endl;
	string readFormat;
	iof->readString(name);
	iof->readString(readFormat);
	// for test
	//cout << "JointNgramWordTranslationModel::read name: " << name << endl;
	iof->readInt(this->modelNumber);
	// for test
	//cout << "JointNgramWordTranslationModel::read modelNumber: " << modelNumber << endl;
	this->allocation();
	// for test
	//cout << "JointNgramWordTranslationModel::read here 1" << endl;
	for (int i = 0; i < modelNumber; i ++) {
		// for test
		//cout << "JointNgramWordTranslationModel::read i: " << i << endl;
		models[i]->read(iof, allocation, blockSize);
	}
	// for test
	//cout << "JointNgramWordTranslationModel::read here 2" << endl;
	this->jointWeightLkt.resize(models[0]->baseNetwork->lkt->weight);
	this->jointWeightLkt.read(iof);
	// for test
	//cout << "JointNgramWordTranslationModel::read here 3" << endl;

	// very important, shared weights
	for (int i = 0; i < modelNumber; i ++) {
		// for test
		//cout << "JointNgramWordTranslationModel::read i: " << i << endl;
		models[i]->baseNetwork->lkt->shareWeight(jointWeightLkt);
		// for test
		//cout << "JointNgramWordTranslationModel::read here 4" << endl;
		models[i]->otl = this->otl;
	}
}
void
JointNgramWordTranslationModel::write(ioFile* iof, int closeFile) {
	// for test
	//cout << "JointNgramWordTranslationModel::write here" << endl;
	iof->writeString(name);
	iof->writeString(iof->format);
	// for test
	//cout << "JointNgramWordTranslationModel::write name: " << name << endl;
	iof->writeInt(this->modelNumber);
	// for test
	//cout << "JointNgramWordTranslationModel::write modelNumber: " << modelNumber << endl;
	for (int i = 0; i < modelNumber; i ++) {
		models[i]->write(iof, 0);
		// for test
		//cout << "JointNgramWordTranslationModel::write i: " << i << endl;
	}
	// for test
	//cout << "JointNgramWordTranslationModel::write here 0.1" << endl;
	//cout << "JointNgramWordTranslationModel::write jointWeightLkt: " << endl;
	//jointWeightLkt.info();
	//cout << "JointNgramWordTranslationModel::write writing matrix: " << endl;
	//jointWeightLkt.write();
	this->jointWeightLkt.write(iof);
	// for test
	//cout << "JointNgramWordTranslationModel::write here 1" << endl;
	if (closeFile == 1) {
		iof->freeWriteFile();
	}
}

int
JointNgramWordTranslationModel::train(char** dataFileName, int* maxExampleNumber, int iteration, string learningRateType, float learningRate, float learningRateDecay) {
	firstTime();
	ioFile dataIof[modelNumber];
	int ngramNumber[modelNumber];
	int N[modelNumber];
	int nm[modelNumber];
	int sumMaxExampleNumber = 0;
	for (int i = 0; i < modelNumber; i ++) {
		dataIof[i].takeReadFile(dataFileName[i]);
		dataIof[i].readInt(ngramNumber[i]);
		dataIof[i].readInt(N[i]);
		NgramWordTranslationModel* model_casted;
		if ((model_casted = (NgramWordTranslationModel*)(models[i])) && N[i] < model_casted->nm) {
			cerr << "ERROR: N in data is wrong:" << N << " < " << nm << endl;
			exit(1);
		}
		nm[i] = model_casted->nm;
		if (maxExampleNumber[i] > ngramNumber[i] || maxExampleNumber[i] == 0) {
			maxExampleNumber[i] = ngramNumber[i];
		}
		sumMaxExampleNumber+=maxExampleNumber[i];
	}
	float currentLearningRate;
	int nstep;
	nstep = sumMaxExampleNumber * (iteration - 1);
	intTensor readTensors[modelNumber];
	intTensor contexts[modelNumber];
	intTensor words[modelNumber];
	floatTensor coefTensor[modelNumber];
	for (int i = 0; i < modelNumber; i ++) {
		readTensors[i].resize(models[i]->blockSize, N[i]);
		contexts[i].sub(readTensors[i], 0, models[i]->blockSize - 1, N[i] - nm[i], N[i] - 2);
		contexts[i].t();
		words[i].select(readTensors[i], 1, N[i] - 1);
		coefTensor[i].resize(models[i]->blockSize, 1);
	}
	int currentExampleNumber = 0;
	int percent = 1;
	float aPercent = sumMaxExampleNumber * CONSTPRINT;
	float iPercent = aPercent * percent;
	int blockNumbers[modelNumber];
	int remainingNumbers[modelNumber];
	int maxBlockNumber = 0;
	for (int i = 0; i < modelNumber; i ++) {
		blockNumbers[i] = maxExampleNumber[i] / models[i]->blockSize;
		if (maxBlockNumber < blockNumbers[i]) {
			maxBlockNumber = blockNumbers[i];
		}
		remainingNumbers[i] = maxExampleNumber[i] - models[i]->blockSize * blockNumbers[i];
	}
	int i;
	cout << sumMaxExampleNumber << " examples" << endl;
	for (i = 0; i < maxBlockNumber; i++) {
		for (int j = 0; j < modelNumber; j ++) {
			if (i < blockNumbers[j]) {
				models[i]->readStripInt(dataIof[j], readTensors[j], coefTensor[j]); // read file n-gram for word and context
				if (dataIof[j].getEOF()) {
					break;
				}
				currentExampleNumber += models[j]->blockSize;
				currentLearningRate = models[j]->takeCurrentLearningRate(learningRate, learningRateType, nstep, learningRateDecay);
				models[j]->trainOne(contexts[j], words[j], coefTensor[j], currentLearningRate, models[j]->blockSize);
				nstep += models[j]->blockSize;
			}
		}
		#if PRINT_DEBUG
			if (currentExampleNumber > iPercent) {
				percent++;
				iPercent = aPercent * percent;
				cout << (float) currentExampleNumber / sumMaxExampleNumber << " ... "
					 << flush;
			}
		#endif
	}
	for (int j = 0; j < modelNumber; j ++) {
		if (remainingNumbers[j] != 0 && !dataIof[j].getEOF()) {
			contexts[j] = 0;
			words[j] = SIGN_NOT_WORD;
			intTensor lastReadTensor(remainingNumbers[j], N[j]);
			models[j]->readStripInt(dataIof[j], lastReadTensor, coefTensor[j]);
			intTensor subReadTensor;
			subReadTensor.sub(readTensors[j], 0, remainingNumbers[j] - 1, 0, N[j] - 1);
			subReadTensor.copy(lastReadTensor);
			if (!dataIof[j].getEOF()) {
				currentLearningRate = models[j]->takeCurrentLearningRate(learningRate, learningRateType, nstep, learningRateDecay);
				models[j]->trainOne(contexts[j], words[j], coefTensor[j], currentLearningRate, remainingNumbers[j]);
			}
		}
	}
	#if PRINT_DEBUG
		cout << endl;
	#endif
	return 1;
}
