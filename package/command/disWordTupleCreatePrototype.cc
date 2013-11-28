#include "mainModel.H"
int getHiddenCode(char* input, intTensor& outputTensor) {
	char hidden[260];
	strcpy(hidden, input);
	int hiddenNumber = 1;
	for (int i = 0; i < strlen(input); i++) {
		if (hidden[i] == '_') {
			hidden[i] = ' ';
			hiddenNumber++;
		}
	}
	string strHidden = hidden;
	istringstream streamHidden(strHidden);
	string word;
	outputTensor.resize(hiddenNumber, 1);
	hiddenNumber = 0;
	while (streamHidden >> word) {
		outputTensor(hiddenNumber) = atoi(word.c_str());
		hiddenNumber++;
	}
	return 1;
}
int main(int argc, char *argv[]) {
	//Create model
	if (argc != 14) {
		cout
				<< "type ngramType inputVocFileName outputVocFileName mapIUnk mapOUnk "
				<< " n dimensionSize nonLinearType"
				<< " hiddenLayerSizeCode codeWordFileName outputNetworkSizeFileName outputModelFileName"
				<< endl;
		cout << "type = dwtovn , dwtovn_nce" << endl;

		return 1;
	} else {
		srand48 (time(NULL));srand(time(NULL));
		NeuralModel* modelPrototype;
		string name = argv[1];
		if (name != WTOVN && name != WTOVN_NCE) {
			cerr << "Which model do you want?" << endl;
			return 1;
		}
		int ngramType = atoi(argv[2]);
		char* contextVocFileName = argv[3];
		char* predictVocFileName = argv[4];
		int mapIUnk = atoi(argv[5]);
		int mapOUnk = atoi(argv[6]);
		int n = atoi(argv[7]);
		int BOS = n;
		int dimensionSize = atoi(argv[8]);

		string nonLinearType = argv[9];
		if (nonLinearType != TANH && nonLinearType != SIGM && nonLinearType
				!= LINEAR) {
			cerr << "Which activation do you want?" << endl;
			return 1;
		}

		char* hiddenLayerSizeCode = argv[10];
		char* codeWordFileName = argv[11];
		char* outputNetworkSizeFileName = argv[12];
		char* outputModelFileName = argv[13];
		int blockSize = 1;
		ioFile iof;
		if (!iof.check(contextVocFileName, 1))
		{
			return 1;
		}
		if (!iof.check(predictVocFileName, 1))
		{
			return 1;
		}
		if (strcmp(codeWordFileName, "xxx"))
		{
			if (!iof.check(codeWordFileName, 1))
			{
				return 1;
			}
		}
		if (strcmp(outputNetworkSizeFileName, "xxx"))
		{
			if (!iof.check(outputNetworkSizeFileName, 1))
			{
				return 1;
			}
		}
		if (iof.check(outputModelFileName, 0))
		{
			cerr << "Prototype exists" << endl;
			return 1;
		}
		intTensor hiddenLayerSizeArray;
		getHiddenCode(hiddenLayerSizeCode, hiddenLayerSizeArray);
		modelPrototype = new NgramWordTranslationModel(name, ngramType,
				contextVocFileName, predictVocFileName, mapIUnk, mapOUnk, BOS,
				blockSize, n, dimensionSize, nonLinearType, hiddenLayerSizeArray,
				codeWordFileName, outputNetworkSizeFileName);
		ioFile mIof;
		mIof.takeWriteFile(outputModelFileName);
		modelPrototype->write(&mIof, 1);

		delete modelPrototype;
		return 0;
	}

}

