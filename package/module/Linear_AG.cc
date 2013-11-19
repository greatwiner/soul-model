#include "mainModule.H"

Linear_AG::Linear_AG() {

}

Linear_AG::Linear_AG(int inputSize, int outputSize, int blockSize, outils* otl)
{
	// Initialize parameters
	name = "Linear_AG";
	this->blockSize = blockSize;
	weightDecay = 0;
	weight.resize(inputSize, outputSize);
	gradWeight.resize(weight);
	bias.resize(outputSize, 1);
	gradBias.resize(bias);
	V1col.resize(blockSize, 1);
	V1col = 1;
	gradInput.resize(inputSize, blockSize);
	output.resize(outputSize, blockSize);
	cumulGradWeight=INIT_VALUE_ADAG;
	cumulGradBias=INIT_VALUE_ADAG;

	this->otl = otl;
	reset();
}

Linear_AG::~Linear_AG() {

}

void
Linear_AG::changeBlockSize(int blockSize) {
	// Need to change memory size for some parameters
	this->blockSize = blockSize;
	int inputSize = gradInput.getSize(0);
	//int inputSize = gradInput.size[0];
	int outputSize = output.getSize(0);
	//int outputSize = output.size[0];
	V1col.resize(blockSize, 1);
	V1col = 1;
	gradInput.resize(inputSize, blockSize);
	output.resize(outputSize, blockSize);

}

void
Linear_AG::reset() {
	weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
	bias.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
	cumulGradWeight=INIT_VALUE_ADAG;
	cumulGradBias=INIT_VALUE_ADAG;
}

floatTensor&
Linear_AG::forward(floatTensor& input) {
	this->input = input;
	output = 0;
	// output = block_size bias columns
	output.ger(bias, V1col, 1);

	output.gemm(weight, 'T', input, 'N', 1, 1);
	return output;
}

floatTensor&
Linear_AG::backward(floatTensor& gradOutput)
{
	// Keep gradOutput for later update
	this->gradOutput = gradOutput;

	// gradInput = weight x gradOutput
	gradInput.gemm(weight, 'N', gradOutput, 'N', 1, 0);
	return gradInput;
}

void
Linear_AG::updateParameters(float learningRate) {
	// for test
	//cout << "Linear_AG::updateParameters here" << endl;
	gradWeight.gemm(input, 'N', gradOutput, 'T', 1, 0);
	// for test
	//cout << "Linear_AG::updateParameters here 1" << endl;
	gradBias.gemv(gradOutput, 'N', V1col, 1, 0);

	// for test
	//cout << "Linear_AG::updateParameters here 2" << endl;
	cumulGradWeight+=gradWeight.averageSquare();
	// for test
	//cout << "Linear_AG::updateParameters here 3" << endl;
	cumulGradBias+=gradBias.averageSquare();

	// for test
	//cout << "Linear_AG::updateParameters here 4" << endl;
	// update parameters
	weight.scal(1 - learningRate * weightDecay/sqrt(cumulGradWeight));
	weight.axpy(gradWeight, -learningRate/sqrt(cumulGradWeight));
	bias.axpy(gradBias, -learningRate/sqrt(cumulGradBias));
	// for test
	//cout << "Linear_AG::updateParameters here 5" << endl;
}

float
Linear_AG::distance2(Linear_AG& anotherLinear) {
	floatTensor distMatrix;
	distMatrix.copy(this->weight);
	distMatrix.axpy(anotherLinear.weight, -1);
	float res1 = distMatrix.sumSquared();
	distMatrix.resize(this->bias);
	distMatrix.copy(this->bias);
	distMatrix.axpy(anotherLinear.bias, -1);
	return res1+distMatrix.sumSquared();
}

void
Linear_AG::read(ioFile *iof) {
	iof->readString(name);
	// for test
	//cout << "Linear_AG::read name: " << name << endl;
	weight.read(iof);
	bias.read(iof);
	iof->readFloat(cumulGradWeight);
	iof->readFloat(cumulGradBias);
}
void
Linear_AG::write(ioFile * iof) {
	iof->writeString(name);
	weight.write(iof);
	bias.write(iof);
	iof->writeFloat(cumulGradWeight);
	iof->writeFloat(cumulGradBias);
}
