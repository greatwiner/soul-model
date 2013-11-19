#include "mainModel.H"
#include "ioFile.H"

int
main(int argc, char* argv[]) {
	if (argc != 6) {
		cout << "fileName outputFileName blockSize oldName newName" << endl;
		return 1;
	}
	char* fileName = argv[1];
	char* outputFileName = argv[2];
	NeuralModel* model = new NgramModel();
	ioFile modelFile;
	modelFile.takeReadFile(fileName);
	int blockSize = atoi(argv[3]);
	char* oldName = argv[4];
	char* newName = argv[5];
	// for test
	cout << "ovn2ovnb::main newName: " << newName << endl;
	// for test
	cout << "ovn2ovnb::main here" << endl;
	model->read(&modelFile, 1, blockSize);
	// for test
	// for test
	cout << "ovn2ovnb.cc::main name: " << model->name << endl;
	if (model->name != oldName) {
		cout << "Type is not good, the program will exit" << endl;
		return 1;
	}
	else {
		model->name = newName;
		// for test
		cout << "ovn2ovnb::main model->name: " << model->name << endl;
		// change the names of corresponding modules with AdaGrad
		if (model->name == OVN_AG) {
			model->baseNetwork->lkt->name = "LookupTable_AG";
			for (int i = 0; i < model->baseNetwork->size; i ++) {
				if (Linear* casted_linear = dynamic_cast<Linear*>(model->baseNetwork->modules[i])) {
					casted_linear->name = "Linear_AG";
				}
			}
			for (int i = 0; i < model->outputNetworkNumber; i ++) {
				model->outputNetwork[i]->name = "LinearSoftmax_AG";
			}
		}
	}
	ioFile outputFile;
	outputFile.takeWriteFile(outputFileName);
	// for test
	cout << "ovn2ovnb::main here" << endl;
	model->write(&outputFile, 1);
	delete model;
	// for test
	cout << "ovn2ovnb::main here1" << endl;
	return 0;
}
