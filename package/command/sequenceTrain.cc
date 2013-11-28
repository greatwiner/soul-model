#include "mainModel.H"

int sequenceTrain(char* prefixModel, char* prefixData, int maxExampleNumber,
		char* trainingFileName, char* validationFileName, string validType,
		string learningRateType, int minIteration, int maxIteration) {
	outils otl;
	char inputModelFileName[260];
	char convertStr[260];
	int iteration;
	int gz = 0;
	for (iteration = maxIteration; iteration >= minIteration - 2; iteration--) {
		sprintf(convertStr, "%ld", iteration);
		strcpy(inputModelFileName, prefixModel);
		strcat(inputModelFileName, convertStr);
		ioFile iof;
		if (!iof.check(inputModelFileName, 0)) {
			strcat(inputModelFileName, ".gz");
			if (iof.check(inputModelFileName, 0)) {
				gz = 1;
				break;
			}
		} else {
			gz = 0;
			break;
		}
	}
	// for test
	//cout << "sequenceTrain::main here 0.1" << endl;
	if (iteration == minIteration - 3) {
		cerr << "Can not find training model " << minIteration - 1 << endl;
		return 1;
	} else if (iteration == maxIteration) {
		cerr << "All is done" << endl;
		return 1;
	}
	// for test
	//cout << "sequenceTrain::main here 0.2" << endl;
	sprintf(convertStr, "%ld", iteration);
	strcpy(inputModelFileName, prefixModel);
	strcat(inputModelFileName, convertStr);
	if (gz) {
		strcat(inputModelFileName, ".gz");
	}
	// for test
	//cout << "sequenceTrain::main here 0.3" << endl;
	ioFile file;
	string name = file.recognition(inputModelFileName);
	// for test
	//cout << "sequenceTrain::main here 0.4" << endl;
	if (name == JWTOVN) {
		MultiplesNeuralModel* model;
		// for test
		//cout << "sequenceTrain::main here 0.5" << endl;
		READMODEL_MULTIPLE(model, 0, inputModelFileName);
		// for test
		//cout << "sequenceTrain::main here 0.6" << endl;
		model->sequenceTrain(prefixModel, gz, prefixData, maxExampleNumber,
				trainingFileName, validationFileName, validType,
				learningRateType, iteration + 1, maxIteration);
		// for test
		//cout << "sequenceTrain::main here 0.7" << endl;
		delete model;
		// for test
		//cout << "sequenceTrain::main here 0.8" << endl;
		return 0;
	} else {
		NeuralModel* model;
		// for test
		//cout << "sequenceTrain::main here" << endl;
		READMODEL(model, 0, inputModelFileName);
		// for test
		//cout << "sequenceTrain::main here1" << endl;

		model->sequenceTrain(prefixModel, gz, prefixData, maxExampleNumber,
				trainingFileName, validationFileName, validType,
				learningRateType, iteration + 1, maxIteration);
		//for test
		//cout << "sequenceTrain::main here2" << endl;
		delete model;
		return 0;
	}
}

int main(int argc, char *argv[]) {
	if (argc != 10) {
		cout
				<< "prefixModel prefixData maxExampleNumber trainingFileName, validationFileName validType learningRateType minIteration maxIteration"
				<< endl;
		cout << "validType: n(normal-text), l(ngram list), id (binary id ngram)"
				<< endl;
		return 0;
	}
	char* prefixModel = argv[1];
	char* prefixData = argv[2];
	int maxExampleNumber = atoi(argv[3]);
	char* trainingFileName = argv[4];
	char* validationFileName = argv[5];
	string validType = argv[6];
	if (validType != "n" && validType != "l" && validType != "id")

	{
		cerr << "Which validType do you want?" << endl;
		return 1;
	}

	string learningRateType = argv[7];
	if (learningRateType != LEARNINGRATE_NORMAL
			&& learningRateType != LEARNINGRATE_DOWN
			&& learningRateType != LEARNINGRATE_ADJUST) {
		cerr << "Which learningRateType do you want?" << endl;
		return 1;
	}
	int minIteration = atoi(argv[8]);
	int maxIteration = atoi(argv[9]);
	srand (time(NULL));sequenceTrain
	(prefixModel, prefixData, maxExampleNumber, trainingFileName,
			validationFileName, validType, learningRateType, minIteration,
			maxIteration);
	return 0;
}

