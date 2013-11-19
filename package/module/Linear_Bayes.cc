#include "mainModule.H"

Linear_Bayes::Linear_Bayes(int inputSize, int outputSize, int blockSize, outils* otl)
: Linear(inputSize, outputSize, blockSize, otl)
{
  //this->prevWeight.resize(inputSize, outputSize);
  //this->gradWeight.resize(this->weight);
  //this->prevGradWeight.resize(this->weight);
  //this->prevBias.resize(bias);
  //this->gradBias.resize(bias);
  //this->prevGradBias.resize(bias);
  pWeight.resize(weight);
  pBias.resize(bias);

  //prevWeight.copy(weight);
  //gradWeight = 0;
  //prevGradWeight = 0;
  //gradBias = 0;
  //prevGradBias = 0;
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
	//this->prevWeight.resize(inputSize, outputSize);
	//this->gradWeight.resize(this->weight);
	//this->prevGradWeight.resize(this->weight);
	bias.resize(outputSize, 1);
	//this->prevBias.resize(bias);
	//this->gradBias.resize(bias);
	//this->prevGradBias.resize(bias);
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
}

floatTensor&
Linear_Bayes::backward(floatTensor& gradOutput, int last)
{
	Linear::backward(gradOutput);

	// accumulate gradients
	//gradWeight.gemm(input, 'N', gradOutput, 'T', 1, 1);
	//gradWeight.axpy(weight, weightDecay);
	//gradBias.gemv(gradOutput, 'N', V1col, 1, 1);
	return gradInput;
}

void
Linear_Bayes::updateParameters(float learningRateForRd, float learningRateForParas, int last)
{
	if (last==1) {
		//this->prevWeight.copy(weight);
		weight.axpy(pWeight, sqrt(learningRateForParas));
		bias.axpy(pBias, sqrt(learningRateForParas));
	}
	else {
		weight.gemm(input, 'N', gradOutput, 'T', -sqrt(learningRateForRd*learningRateForParas),
		      1 - sqrt(learningRateForRd*learningRateForParas) * weightDecay);
		bias.gemv(gradOutput, 'N', V1col, -sqrt(learningRateForRd*learningRateForParas), 1);
	}
	/*if (weight.testNan() != 0) {
		cout << "Linear_Bayes::updateParameters weight is nan, last=" << last << endl;
		ioFile file1, file2, file3, file4;
		file1.takeWriteFile("/vol/work/dokhanh/wmt13/esLM/0/10gram/pWeight");
		file2.takeWriteFile("/vol/work/dokhanh/wmt13/esLM/0/10gram/pBias");
		file3.takeWriteFile("/vol/work/dokhanh/wmt13/esLM/0/10gram/input");
		file4.takeWriteFile("/vol/work/dokhanh/wmt13/esLM/0/10gram/gradOutput");
		cout << "Linear_Bayes::updateParameters test " << input.testInf() << endl;
		if (last==1) {
			pWeight.write(&file1);
			pBias.write(&file2);
		}
		else {
			input.write(&file3);
			gradOutput.write(&file4);
		}
		exit(0);
	}*/
}

void
Linear_Bayes::updateRandomness(float learningRateForRd) {
	pWeight.gemm(input, 'N', gradOutput, 'T', -sqrt(learningRateForRd), 1);
	pWeight.axpy(weight, -sqrt(learningRateForRd)*weightDecay);
	pBias.gemv(gradOutput, 'N', V1col, -sqrt(learningRateForRd), 1);
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
