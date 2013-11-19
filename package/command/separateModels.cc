#include "mainModel.H"

int
main(int argc, char *argv[]) {
	if (argc != 4) {
		cout << "modelFileName blockSize outputPrefix" << endl;
		return 0;
	}
	char* modelFileName = argv[1];
	ioFile modelFile;
	modelFile.takeReadFile(modelFileName);
	MultiplesNeuralModel* multiplesModel;
	int blockSize = atoi(argv[2]);
	// for test
	cout << "separateModels::main here" << endl;
	READMODEL_MULTIPLE(multiplesModel, blockSize, modelFileName);
	// for test
	cout << "separateModels::main here 1" << endl;
	int modelNumber = multiplesModel->modelNumber;
	char* outputPrefix = argv[3];
	char outputFileName[260];
	char convertStr[260];
	for (int i = 0; i < modelNumber; i ++) {
		// for test
		cout << "separateModels::main i: " << i << endl;
		if (multiplesModel->name == JWTOVN) {
			// copy the shared representations to each model
			multiplesModel->models[i]->baseNetwork->lkt->weight.haveMemory = 1;
		}
		// for test
		cout << "separateModels::main here 2" << endl;
		strcpy(outputFileName, outputPrefix);
		sprintf(convertStr, "%d", i);
		strcat(outputFileName, convertStr);
		ioFile outputFile;
		outputFile.takeWriteFile(outputFileName);
		// for test
		cout << "separateModels::main here 3" << endl;
		multiplesModel->models[i]->write(&outputFile, 1);
		// for test
		cout << "separateModels::main here 4" << endl;
	}
	// for test
	cout << "separateModels::main here 5" << endl;
	for (int i = 0; i < modelNumber; i ++) {
		if (multiplesModel->name == JWTOVN) {
			multiplesModel->models[i]->baseNetwork->lkt->weight.haveMemory = 0;
		}
	}
	delete multiplesModel;
}
