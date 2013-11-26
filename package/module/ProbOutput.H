class ProbOutput : public Module {
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

	ProbOutput();

	//virtual
	//ProbOutput(int inputSize, int outputSize, int blockSize, outils* otl) = 0;
	~ProbOutput() = 0;
	virtual void
	reset() = 0;
	void
	changeBlockSize(int blockSize);
	floatTensor&
	forward(floatTensor& input);
	floatTensor&
	backward(floatTensor& word);
	virtual floatTensor&
	backward(intTensor& word, floatTensor& coefTensor) = 0;
	virtual void
	updateParameters(float learningRate) = 0;
	float
	distance2(Module& anotherOutput);
	virtual void
	read(ioFile *iof) = 0;
	virtual void
	write(ioFile * iof) = 0;
};
