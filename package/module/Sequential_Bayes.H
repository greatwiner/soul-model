/* Specific module used to build a sequential network.  The methods
 forward, backward, and updateParameters only call sequentially the
 corresponding methods of each module. The add method aims to add a
 new module, and the last added module is the output (the first
 added module is the input of the network).
 */

class Sequential_Bayes : public Sequential
{
public:

	Sequential_Bayes(int maxSize);
	~Sequential_Bayes();
  floatTensor&
  backward(floatTensor& gradOutput, int last);
  void
  read(ioFile *iof);
  void
  write(ioFile *iof);

  int
  numberOfWeights();

  float
  sumSquaredWeights();

  void
  initializeP();

  float
  calculeH();

  void
  updateRandomness(float learningRate);

  void
  updateParameters(float learningRateForRd, float learningRateForParas, int last);
};

