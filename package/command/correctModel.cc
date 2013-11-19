#include "mainModel.H"
#include "ioFile.H"

int
main(int argc, char* argv[]) {
	if (argc != 4) {
		cout << "fileName outputFileName blockSize" << endl;
		return 1;
	}
	char* fileName = argv[1];
	char* outputFileName = argv[2];
	NeuralModel* model = new NgramModel();
	ioFile modelFile;
	modelFile.takeReadFile(fileName);
	int blockSize = atoi(argv[3]);
	model->read(&modelFile, 1, blockSize);
	// correct parameters
	model->baseNetwork->lkt->weight.correct(model->otl);
	cout << "lookupTable corrected" << endl;
	ioFile outputFile;
	outputFile.takeWriteFile(outputFileName);
	model->write(&outputFile, 1);
	delete model;
	return 0;
}
