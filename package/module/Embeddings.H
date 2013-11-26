class Embeddings : public Module {
public:
	floatTensor gradWeight;
	intTensor input;
	// tensor used to select some part of other tensors, act as a pointer
	floatTensor selectOutput;
	floatTensor selectWeight;
	floatTensor selectGradWeight;

	// projection dimension
	int dimensionSize;

	// number of indices ~ number of words in the vocabulary
	int indexNumber;

	outils* otl;

	Embeddings();
	Embeddings(int nIndex, int nDimension, int inputSize, int blockSize,
		      int oneClass, outils* otl);

	~Embeddings();

	virtual void
	reset() = 0;

	void
	changeBlockSize(int blockSize);

	floatTensor&
	forward(floatTensor& input);

	floatTensor&
	backward(floatTensor& gradOutput);

	floatTensor&
	forward(intTensor& input);

	virtual void
	updateParameters(float learningRate) = 0;

	float
	distance2(Module& anotherLkt);

	virtual void
	read(ioFile *iof) = 0;

	virtual void
	write(ioFile * iof) = 0;

	// Initialize with only one vector for all words
	void
	init1class();
};
