#include "mainModule.H"
LinearSoftmax_Bayes::LinearSoftmax_Bayes(int inputSize, int outputSize, int blockSize,
    outils* otl) : LinearSoftmax(inputSize, outputSize, blockSize, otl)
{
  //prevWeight.resize(weight);
  //gradWeight.resize(weight);
  //prevGradWeight.resize(weight);
  //prevBias.resize(bias);
  //gradBias.resize(bias);
  //prevGradBias.resize(bias);
  pWeight.resize(weight);
  pBias.resize(bias);

  //prevWeight.copy(weight);
  //gradWeight = 0;
  //prevGradWeight = 0;
  //gradBias = 0;
  //prevGradBias = 0;
}

void
LinearSoftmax_Bayes::changeBlockSize(int blockSize)
{
	this->blockSize = blockSize;
	V1col.resize(blockSize, 1);
	V1col = 1;
	int inputSize = gradInput.size[0];
	int outputSize = output.size[0];
	softmaxVCol.resize(blockSize, 1);
	gradInput.resize(inputSize, blockSize);
	output.resize(outputSize, blockSize);
	gradOutput.resize(output);
	preOutput.resize(outputSize, blockSize);
	//prevWeight.resize(weight);
	//gradWeight.resize(weight);
	//prevGradWeight.resize(weight);
	bias.resize(outputSize, 1);
	//prevBias.resize(bias);
	//gradBias.resize(bias);
	//prevGradBias.resize(bias);
	pWeight.resize(weight);
	pBias.resize(bias);

}

void
LinearSoftmax_Bayes::reset()
{
  weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  bias.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
}

floatTensor&
LinearSoftmax_Bayes::backward(floatTensor& word)
{
  cerr << "ERROR: backward of LinearSoftmax for realTensor" << endl;
  exit(1);
}

floatTensor&
LinearSoftmax_Bayes::backward(intTensor& word, int last) {
    LinearSoftmax::backward(word);
    // accumulate gradient
	//gradWeight.gemm(input, 'N', gradOutput, 'T', 1, 1);
	//gradWeight.axpy(weight, weightDecay);
	//gradBias.gemv(gradOutput, 'N', V1col, 1, 1);
	return gradInput;
}

void
LinearSoftmax_Bayes::updateParameters(float learningRateForRd, float learningRateForParas, int last)
{
	if (last==1) {
		//this->prevWeight.copy(weight);
		//this->prevBias.copy(bias);
		weight.axpy(pWeight, sqrt(learningRateForParas));
		bias.axpy(pBias, sqrt(learningRateForParas));
	}
	else {
		weight.gemm(input, 'N', gradOutput, 'T', -sqrt(learningRateForRd*learningRateForParas),
		      1 - sqrt(learningRateForRd*learningRateForParas) * weightDecay);
		bias.gemv(gradOutput, 'N', V1col, -sqrt(learningRateForRd*learningRateForParas), 1);
	}
	/*if (weight.testNan() != 0) {
		weight.copy(prevWeight);
	}
	if (bias.testNan() != 0) {
		bias.copy(prevBias);
	}*/
}

void
LinearSoftmax_Bayes::updateRandomness(float learningRateForRd) {
	pWeight.gemm(input, 'N', gradOutput, 'T', -sqrt(learningRateForRd), 1);
	pWeight.axpy(weight, -sqrt(learningRateForRd)*weightDecay);
	pBias.gemv(gradOutput, 'N', V1col, -sqrt(learningRateForRd), 1);
}

int
LinearSoftmax_Bayes::numberOfWeights() {
	return weight.size[0]*weight.size[1];
}

float
LinearSoftmax_Bayes::sumSquaredWeights() {
	return weight.sumSquared();
}

void
LinearSoftmax_Bayes::initializeP() {
	this->pWeight.initializeNormal(this->otl);
	this->pBias.initializeNormal(this->otl);
}

float
LinearSoftmax_Bayes::getKinetic() {
	ki = 0.5*(this->pWeight.sumSquared() + this->pBias.sumSquared());
	return ki;
}

float
LinearSoftmax_Bayes::getWeightDecayTerm() {
	wD = 0.5*weight.sumSquared();
	return wD;
}

float
LinearSoftmax_Bayes::calculeH() {
	ki = this->getKinetic();
	wD = this->getWeightDecayTerm();
	// for test
	//cout << "LinearSoftmax_Bayes::calculeH kinetic: " << ki << endl;
	//cout << "LinearSoftmax_Bayes::calculeH weight decay: " << wD << endl;
	return ki + this->weightDecay*wD;
}
