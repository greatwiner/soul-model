class JointNgramWordTranslationModel : public MultiplesNeuralModel {
public:
	floatTensor jointWeightLkt;
	int whereInitCommonWeight;

	JointNgramWordTranslationModel();
	~JointNgramWordTranslationModel();

	void
	allocation();

	// create a joint model from separate models
	// whereInitCommonWeight: we take
	JointNgramWordTranslationModel(NgramWordTranslationModel** transModels, int modelNumber, int whereInitCommonWeight);

	// Read, write with file
	void
	read(ioFile* iof, int allocation, int blockSize);
	void
	write(ioFile* iof, int closeFile);

	int
	train(char** dataFileName, int* maxExampleNumber, int iteration, string learningRateType, float learningRate, float learningRateDecay);
};
