#include "mainModel.H"
#include "ioFile.H"

int
main(int argc, char *argv[]) {
	if (argc < 5) {
		cout << "modelNumber modelFileName1... blockSize modelFileOutName" << endl;
		return 0;
	}
	int modelNumber = atoi(argv[1]);
	cout << "create a joint model from " << modelNumber << " sub-models" << endl;
	char* modelFileNames[modelNumber];
	NgramWordTranslationModel* models[modelNumber];
	ioFile ioFiles[modelNumber];
	int blockSize = atoi(argv[2 + modelNumber]);
	for (int i = 0; i < modelNumber; i++) {
		modelFileNames[i] = argv[i+2];
		models[i] = new NgramWordTranslationModel();
		ioFiles[i].takeReadFile(modelFileNames[i]);
		models[i]->read(&ioFiles[i], 1, blockSize);
	}
	// for test
	//cout << "createJointWordTranslationModel::main reference: " << modelArray[0]->outputVoc->table[0]->next->next << " index: " << modelArray[0]->outputVoc->table[0]->next->next->index << " word: " << modelArray[0]->outputVoc->table[0]->next->next->word << endl;
	JointNgramWordTranslationModel* model = new JointNgramWordTranslationModel(models, modelNumber, 0);
	// for test
	//cout << "createJointWordTranslationModel::main here 3" << endl;
	// for test
	//cout << "createJointWordTranslationModel::main reference: " << modelArray[0]->outputVoc->table[0]->next->next << " index: " << modelArray[0]->outputVoc->table[0]->next->next->index << " word: " << modelArray[0]->outputVoc->table[0]->next->next->word << endl;

	ioFile modelFileOut;
	modelFileOut.takeWriteFile(argv[3 + modelNumber]);
	model->write(&modelFileOut, 1);
	delete model;
	// for test
	//cout << "createJointWordTranslationModel::main here 7" << endl;
}
