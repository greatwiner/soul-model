/* 
 Specific class for language model. This Layer aims to be the uppest
 layer of the network. Basically, this code represent a Linear
 Module join with a softmax activation. See the Linear class for
 more details.
 */

class LinearSoftmax_Bayes : public LinearSoftmax
{
public:

	floatTensor pWeight;
	floatTensor pBias;

	// kinetic energy
	float ki;

	// weight decay term
	float wD;

	LinearSoftmax_Bayes(int inputSize, int outputSize, int blockSize, outils* otl);

  void
  reset();
  void
  changeBlockSize(int blockSize);
  floatTensor&
  backward(floatTensor& word);
  floatTensor&
  backward(intTensor& word, int last);
  void
  updateParameters(float learningRateForRd, float learningRateForParas, int last);

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
  updateRandomness(float learningRate);
};

