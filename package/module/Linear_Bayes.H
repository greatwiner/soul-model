/* Linear Layer. May be plugged with a non-linear activity function,
 such as softmax of tanh.
 */

class Linear_Bayes : public Linear
{
public:
	floatTensor prevWeight;
	floatTensor gradWeight;
	floatTensor prevGradWeight;
	floatTensor prevBias;
	floatTensor gradBias;
	floatTensor prevGradBias;
	floatTensor pWeight;
	floatTensor pBias;

	// kinetic energy
	float ki;

	// weight decay term
	float wD;

	Linear_Bayes(int inputSize, int outputSize, int blockSize, outils* otl);
  void
  updateParameters(float learningRate);

  void
  changeBlockSize(int blockSize);

  void
  reset();

  floatTensor&
  backward(floatTensor& gradOutput, int last);

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
  updateRandomness(float learningRate);
};

