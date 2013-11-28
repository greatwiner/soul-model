class NgramNCEWordTranslationDataSet : public DataSet {
public:
	// ngramType 0: trgSrc, 1: trg, 2: srcTrg, 3: src
	int nm;
	// coefficients for each n-gram
	float* coef;
	//From DataSet
	NgramNCEWordTranslationDataSet(int type, int n, int BOS, SoulVocab* inputVoc, SoulVocab* outputVoc, int mapIUnk, int mapOUnk, int maxNgramNumber);

	~NgramNCEWordTranslationDataSet();

	int
	resamplingSentence(int totalLineNumber, int resamplingLineNumber, int* resamplingLineId);
	int
	readText(ioFile* iof);
	int
	resamplingText(ioFile* iof, int totalLineNumber, int resamplingLineNumber);
	// the text does not contain coefficients; we have to randomly generate a negative example for each positive one

	intTensor&
	createTensor();

	// this function add a line and a negative example corresponding to this line
	int
	addLine(string line);
	// add line and a coefficient corresponding to this line. If the coefficient < 0, this line contains negative examples
	int
	addLine(ioFile* iof);
	int
	readTextNgram(ioFile* iof);

	void
	writeReBiNgram(ioFile* iof);

	int
	readCoBiNgram(ioFile* iof);

	//Internal function:

	int
	addDisWordTuple(int* srcIndex, int* desIndex, int* unkIndex);

	void
	shuffle(int times);

	void
	sortNgram();

	int
	writeReBiNgram();
	float
	computePerplexity();

	// grouping n-grams by summing coefficients
	int
	groupingNgram();
};
