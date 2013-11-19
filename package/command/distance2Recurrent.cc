#include "mainModel.H"

int
main(int argc, char *argv[]) {
	char* fileName1 = argv[1];
	char* fileName2 = argv[2];
	// for test
	//cout << "distance2Recurrent::main here" << endl;
	ioFile modelFile1;
	modelFile1.takeReadFile(fileName1);
	ioFile modelFile2;
	modelFile2.takeReadFile(fileName2);
	// for test
	//cout << "distance2Recurrent::main here 1" << endl;
	RecurrentModel* model1 = new RecurrentModel();
	RecurrentModel* model2 = new RecurrentModel();
	int blockSize = 1; // by default
	// for test
	//cout << "distance2Recurrent::main here 4" << endl;
	dynamic_cast<RecurrentModel*>(model1)->read(&modelFile1, 1, blockSize);
	// for test
	//cout << "distance2Recurrent::main here 2" << endl;
	dynamic_cast<RecurrentModel*>(model2)->read(&modelFile2, 1, blockSize);
	// for test
	//cout << "distance2Recurrent::main here 3" << endl;
	cout << "distance2Recurrent::main dist: " << dynamic_cast<RecurrentModel*>(model1)->distance2(*(dynamic_cast<RecurrentModel*>(model2))) << endl;
	delete model1;
	delete model2;
}
