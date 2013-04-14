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

	// kinetic energy
	float ki;

	// weight decay term;
	float wD;

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
  getKinetic();

  float
  getWeightDecayTerm();

  float
  calculeH();

  void
  reUpdateParameters(int accept);

  void
  updateRandomness(float learningRate, float RATE);
};

