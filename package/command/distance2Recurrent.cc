#include "mainModel.H"

int main(int argc, char *argv[]) {
	if (argc != 4) {
		cout << "fileName1 fileName2 modelType" << endl;
		exit(1);
	}
	char* fileName1 = argv[1];
	char* fileName2 = argv[2];
	string modelType = argv[3];
	// for test
	//cout << "distance2Recurrent::main here" << endl;
	ioFile modelFile1;
	modelFile1.takeReadFile(fileName1);
	ioFile modelFile2;
	modelFile2.takeReadFile(fileName2);
	// for test
	//cout << "distance2Recurrent::main here 1" << endl;
	/*RecurrentModel* model1 = new RecurrentModel();
	 RecurrentModel* model2 = new RecurrentModel();*/
	NeuralModel* model1;
	NeuralModel* model2;
	if (modelType == WTOVN || modelType == WTOVN_NCE) {
		model1 = new NgramWordTranslationModel();
		model2 = new NgramWordTranslationModel();
	} else {
		model1 = new NgramModel();
		model2 = new NgramModel();
	}
	int blockSize = 1; // by default
	// for test
	//cout << "distance2Recurrent::main here 4" << endl;
	model1->read(&modelFile1, 1, blockSize);
	// for test
	//cout << "distance2Recurrent::main here 2" << endl;
	model2->read(&modelFile2, 1, blockSize);
	// for test
	//cout << "distance2Recurrent::main here 3" << endl;
	cout << "distance2Recurrent::main dist: " << model1->distance2(*model2)
			<< endl;
	delete model1;
	delete model2;
}
