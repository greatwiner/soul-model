class Linear_AG : public Module {
public:
	// Pointer for the output of the previous layer, used for updateParameters
	floatTensor input;

	// For the bunch mode, copy bias vector for all columns of
	// (floatTensor) output using matrix multiplication
	floatTensor V1col;

	// Pointer to a random generator
	outils* otl;

	float cumulGradWeight;
	float cumulGradBias;
	floatTensor gradWeight;
	floatTensor gradBias;

	Linear_AG();

	Linear_AG(int inputSize, int outputSize, int blockSize, outils* otl);

	~Linear_AG();

	void
	reset();

	void
	changeBlockSize(int blockSize);

	floatTensor&
	forward(floatTensor& input);

	floatTensor&
	backward(floatTensor& gradOutput);

	void
	updateParameters(float learningRate);

	float
	distance2(Module& anotherLinear);

	void
	read(ioFile *iof);
	void
	write(ioFile * iof);
};
