class LinearSoftmax_NCE : public ProbOutput {
public:
	LinearSoftmax_NCE();
	LinearSoftmax_NCE(int inputSize, int outputSize, int blockSize, outils* otl);
	~LinearSoftmax_NCE();
	void
	reset();
	floatTensor&
	backward(intTensor& word, floatTensor& coefTensor);
	void
	updateParameters(float learningRate);
	void
	read(ioFile *iof);
	void
	write(ioFile * iof);
};
