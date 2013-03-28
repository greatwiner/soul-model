#include "mainModule.H"

Linear_Bayes::Linear_Bayes(int inputSize, int outputSize, int blockSize, outils* otl)
: Linear(inputSize, outputSize, blockSize, otl)
{
  this->prevWeight.resize(inputSize, outputSize);
  this->gradWeight.resize(this->weight);
  this->prevGradWeight.resize(this->weight);
  this->prevBias.resize(bias);
  this->gradBias.resize(bias);
  this->prevGradBias.resize(bias);
  pWeight.resize(weight);
  pBias.resize(bias);

  prevWeight.copy(weight);
  gradWeight = 0;
  prevGradWeight = 0;
  gradBias = 0;
  prevGradBias = 0;
}

void
Linear_Bayes::changeBlockSize(int blockSize)
{
	this->blockSize = blockSize;
	int inputSize = gradInput.size[0];
	int outputSize = output.size[0];
	V1col.resize(blockSize, 1);
	V1col = 1;
	gradInput.resize(inputSize, blockSize);
	output.resize(outputSize, blockSize);
	this->prevWeight.resize(inputSize, outputSize);
	this->gradWeight.resize(this->weight);
	this->prevGradWeight.resize(this->weight);
	bias.resize(outputSize, 1);
	this->prevBias.resize(bias);
	this->gradBias.resize(bias);
	this->prevGradBias.resize(bias);
	gradInput.resize(inputSize, blockSize);
	output.resize(outputSize, blockSize);
	pWeight.resize(weight);
	pBias.resize(bias);
}

void
Linear_Bayes::reset()
{
  weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  bias.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
  prevWeight.copy(weight);
  gradWeight = 0;
  prevGradWeight = 0;
  gradBias = 0;
  prevGradBias = 0;
}

floatTensor&
Linear_Bayes::backward(floatTensor& gradOutput, int last)
{
	Linear::backward(gradOutput);

	// accumulate gradients
	gradWeight.gemm(input, 'N', gradOutput, 'T', 1, 1);
	if (last == 1) {
		gradWeight.axpy(weight, weightDecay);
	}
	gradBias.gemv(gradOutput, 'N', V1col, 1, 1);
	return gradInput;
}

void
Linear_Bayes::updateParameters(float learningRate)
{
  /*weight.gemm(input, 'N', gradOutput, 'T', -learningRate,
      1 - learningRate * weightDecay);
  bias.gemv(gradOutput, 'N', V1col, -learningRate, 1);*/
	weight.axpy(pWeight, sqrt(2*learningRate));
	bias.axpy(pBias, sqrt(2*learningRate));

}

void
Linear_Bayes::updateRandomness(float learningRate) {
	pWeight.axpy(gradWeight, -sqrt(0.5*learningRate));
	pBias.axpy(gradBias, -sqrt(0.5*learningRate));
}

int
Linear_Bayes::numberOfWeights() {
	return weight.size[0]*weight.size[1];
}

float
Linear_Bayes::sumSquaredWeights() {
	return weight.sumSquared();
}

void
Linear_Bayes::initializeP() {
	this->pWeight.initializeNormal(this->otl);
	this->pBias.initializeNormal(this->otl);
}

float
Linear_Bayes::getKinetic() {
	ki = 0.5*(this->pWeight.sumSquared() + this->pBias.sumSquared());
	return ki;
}

float
Linear_Bayes::getWeightDecayTerm() {
	wD = 0.5*weight.sumSquared();
}

float
Linear_Bayes::calculeH() {
	ki = this->getKinetic();
	wD = this->getWeightDecayTerm();
	// for test
	cout << "Linear_Bayes::calculeH kinetic: " << ki << endl;
	cout << "Linear_Bayes::calculeH weight decay: " << wD << endl;
	return ki + this->weightDecay*wD;
}

void
Linear_Bayes::reUpdateParameters(int accept) {
	if (accept == 1) {
		prevWeight.copy(weight);
		prevBias.copy(bias);
		prevGradWeight.copy(gradWeight);
		prevGradBias.copy(gradBias);
	}
	else {
		// for test
		//cout << "vong 1" << endl;
		weight.copy(prevWeight);
		bias.copy(prevBias);
		// for test

		gradWeight.copy(prevGradWeight);
		// for test
		//cout << "vong 3" << endl;
		gradBias.copy(prevGradBias);
		// for test
		//cout << "vong 4" << endl;
	}
}
