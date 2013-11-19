class LinearSoftmax_AG : public Module {
public:
	/*same as Linear */
	floatTensor gradOutput;
	floatTensor input;
	floatTensor softmaxVCol;
	floatTensor V1col;
	floatTensor softmaxV1row;

	/* Specific, only used for bunch mode computation*/
	floatTensor selectGradOutput;
	floatTensor preOutput;
	floatTensor selectOutput;
	floatTensor selectPreOutput;
	outils* otl;

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

	void
	changeBlockSize(int blockSize);

	floatTensor&
	forward(floatTensor& input);

	floatTensor&
	backward(floatTensor& word);

	floatTensor&
	backward(intTensor& word);

	void
	updateParameters(float learningRate);

	float
	distance2(LinearSoftmax& anotherOutput);

	void
	read(ioFile *iof);
	void
	write(ioFile * iof);
};
