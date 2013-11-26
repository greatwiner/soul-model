class NgramNCEDataSet : public DataSet {
public:
	int lengthPerNgram;
	int realNgramNumberAfterGrouping;

	// coefficient for each ngram;
	float* coef;

	NgramNCEDataSet(int type, int n, int BOS, SoulVocab* inputVoc,
	      SoulVocab* outputVoc, int mapIUnk, int mapOUnk, int maxNgramNumber);

	int
	setMaxNgramNumber(int maxNgramNumber);

	int
	addLine(string line);

	int
	addLineWithCoef(string line, float coefficient);

	int
	addLine(ioFile* iof);
	int
	resamplingSentence(int totalLineNumber, int resamplingLineNumber,
	int* resamplingLineId);
	int
	readTextNoCoef(ioFile* iof);
	int
	readText(ioFile* iof);
	int
	resamplingTextNoCoef(ioFile* iof, int totalLineNumber, int resamplingLineNumber);
	int
	resamplingText(ioFile* iof, int totalLineNumber, int resamplingLineNumber);

	intTensor&
	createTensor();

	int
	readTextNgram(ioFile* iof);
	// the same readTextNgram, but with a coefficient at the end of each line
	int
	readTextNgramWithCoef(ioFile* iof);

	void
	writeReBiNgram(ioFile* iof);

	int
	readCoBiNgram(ioFile* iof);

	//Other functions
	void
	shuffle(int times);
	// grouping same n-gram in adding their coefficients, return the number of different n-grams
	int
	groupingNgram();

	void
	sortNgram();
	string
	inverse(string line);
	int
	writeReBiNgram();
	float
	computePerplexity();
};
