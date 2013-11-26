class LinearSoftmax_AG : public ProbOutput {
public:
	// specific for AdaGrad method
	float cumulGradWeight;
	float cumulGradBias;
	floatTensor gradWeight;
	floatTensor gradBias;

	LinearSoftmax_AG();

	LinearSoftmax_AG(int inputSize, int outputSize, int blockSize, outils* otl);

	~LinearSoftmax_AG();

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
