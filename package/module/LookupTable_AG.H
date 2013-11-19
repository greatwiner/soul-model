

class LookupTable_AG : public Embeddings {
public:

	floatTensor cumulGradWeight;

	LookupTable_AG();

	LookupTable_AG(int nIndex, int nDimension, int inputSize, int blockSize,
		      int oneClass, outils* otl);

	~LookupTable_AG();

	void
	reset();

	void
	updateParameters(float learningRate);

	void
	read(ioFile *iof);
	void
	write(ioFile * iof);
};
