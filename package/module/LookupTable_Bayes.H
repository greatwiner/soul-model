/*
 * A lookup table: in updateParameters only the involved vectors are
 * updated.
 */

class LookupTable_Bayes : public LookupTable
{
public:
	floatTensor prevWeight;
	floatTensor prevGradWeight;
	floatTensor pWeight;

  LookupTable_Bayes(int nIndex, int nDimension, int inputSize,
      int blockSize, int oneClass, outils* otl);

  ~LookupTable_Bayes();
  void
  reset();
  void
  changeBlockSize(int blockSize);
  floatTensor&
  backward(floatTensor& gradOutput, int last);
  void
  updateParameters(float learningRate);

  void
  init1class();

  int
  numberOfWeights();

  float
  sumSquaredWeights();

  void
  initializeP();

  float
  calculeH();

  void
  reUpdateParameters(int accept);

  void
  updateRandomness(float learningRate);
};

