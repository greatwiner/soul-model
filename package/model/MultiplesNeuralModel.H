// training simultaneously multiples neural networks, by sharing some parameters
class MultiplesNeuralModel {
public:
	string name;
	int modelNumber;
	NeuralModel** models;

	// this outils will be shared by all models, (I suppose)
	outils* otl;

	MultiplesNeuralModel();
	MultiplesNeuralModel(int modelNumber);
	virtual
	~MultiplesNeuralModel();

	// allocation for each model separately
	virtual void
	allocation() = 0;

	int
	decodeWord(int modelIndex, intTensor& word);
	int
	decodeWord(int modelIndex, intTensor& word, int subBlockSize);

	// Edit weights which represent word spaces
	void
	setWeight(int modelIndex, char* layerName, floatTensor& tensor);
	floatTensor&
	getWeight(int modelIndex, char* layerName);
	void
	setWeightDecay(int modelIndex, float weightDecay);

	// Change block size, the number of examples in bunch mode
	void
	changeBlockSize(int blockSize);
	void
	changeBlockSize(int modelIndex, int blockSize);

	floatTensor&
	forwardOne(int modelIndex, intTensor& context, intTensor& word);

	floatTensor&
	computeProbability(int modelIndex, char* textFileName, string textType);

	float
	computePerplexity(int modelIndex, char* textFileName, string textType);

	// Compute probabilities, suppose that n-grams are already read in DataSet
	floatTensor&
	computeProbability(int modelIndex);

	// Compute perplexity, suppose that n-grams are already read in DataSet
	float
	computePerplexity(int modelIndex);

	// Compute probabilites for ngram in ngramTensor, write to probTensor
	int
	forwardProbability(int modelIndex, intTensor& ngramTensor, floatTensor& probTensor);

	// Only for recurrent models
	void
	firstTime(int modelIndex);
	void
	firstTime(int modelIndex, intTensor& context);
	void
	firstTime();

	// trainOne: Train with one example or one block of examples
	// shared with all sub-classes
	void
	trainOne(int modelIndex, intTensor& context, intTensor& word, float learningRate);
	void
	trainOne(int modelIndex, intTensor& context, intTensor& word, float learningRate,
			int subBlockSize);

	// Only for test and debugging with artificial data
	int
	trainTest(int modelIndex, int maxExampleNumber, float weightDecay, string learningRateType,
	      float learningRate, float learningRateDecay, intTensor& gcontext,
	      intTensor& gword);

	// train: Train with one n-gram file (1 epoch)
	int
	train(int modelIndex, char* dataFileName, int maxExampleNumber, int iteration,
	      string learningRateType, float learningRate, float learningRateDecay);

	// train simultaneously multiple models
	virtual int
	train(char** dataFileName, int* maxExampleNumber, int iteration, string learningRateType, float learningRate, float learningRateDecay) = 0;

	// sequenceTrain with several epochs using early stopping,
	// call train(...) for several times
	// maxExampleNumber = 0 means using all examples in resampling data files
	// learningRateType is
	// 'n': normal: learningRate = function of seen examples
	// 'd': down: learningRate is fixed for each epoch, is divided by learningRateDecay
	// if perplexity of dev data increases
	// if prefixModel = 'xxx': Don't write models to file
	int
	sequenceTrain(char* prefixModel, int gz, char* prefixData,
	      int* maxExampleNumber, char* trainingFileName, char* validationFileName, string validType, string learningRateType, int minIteration, int maxIteration);
	int
	sequenceTrain(char* prefixModel, int gz, char* prefixData,
		  int maxExampleNumber, char* trainingFileName, char* validationFileName, string validType, string learningRateType, int minIteration, int maxIteration);

	// Read, write with file
	virtual void
	read(ioFile* iof, int allocation, int blockSize) = 0;
	virtual void
	write(ioFile* iof, int closeFile) = 0;
};
