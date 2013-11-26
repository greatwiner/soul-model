#include "mainModule.H"

LinearSoftmax_AG::LinearSoftmax_AG() {

}

LinearSoftmax_AG::LinearSoftmax_AG(int inputSize, int outputSize, int blockSize,
    outils* otl)
{
	name = "LinearSoftmax_AG";
	this->blockSize = blockSize;
	weightDecay = 0;
	weight.resize(inputSize, outputSize);
	bias.resize(outputSize, 1);
	V1col.resize(blockSize, 1);
	V1col = 1;
	softmaxV1row.resize(outputSize, 1);
	softmaxV1row = 1;
	softmaxVCol.resize(blockSize, 1);
	gradInput.resize(inputSize, blockSize);
	output.resize(outputSize, blockSize);
	gradOutput.resize(output);
	preOutput.resize(outputSize, blockSize);
	this->cumulGradWeight=INIT_VALUE_ADAG;
	this->cumulGradBias=INIT_VALUE_ADAG;
	this->gradWeight.resize(weight);
	this->gradBias.resize(bias);

	this->otl = otl;
	reset();
}

LinearSoftmax_AG::~LinearSoftmax_AG() {

}

void
LinearSoftmax_AG::reset()
{
	weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
	bias.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
	this->cumulGradWeight=INIT_VALUE_ADAG;
	this->cumulGradBias=INIT_VALUE_ADAG;
}

floatTensor&
LinearSoftmax_AG::backward(intTensor& word, floatTensor& coefTensor) {
	// gradOutput for weight and bias of the Linear part,
	// computed from the output after softmax (not the output of
	// the Linear part
	// coefTensor: this tensor is only for NCE algorithm
	gradOutput.copy(output);
	int i;
	this->input = input;
	for (i = 0; i < blockSize; i++) {
		// If taking account this n-gram
		// In some cases, for some examples in the block, we don't
		// want to update with them, e.g., blockSize = 128 but in the
		// last block, we have only 78, predicted word for 50 *
		// last examples should be set to SIGN_NOT_WORD
		// If using, subtract its value in gradOutput 1
		if (word(i) != SIGN_NOT_WORD) {
			gradOutput(word(i), i) -= 1;
        }
		// Not use, all values = 0
		else {
			selectGradOutput.select(gradOutput, 1, i);
			selectGradOutput = 0;
        }
    }
	gradInput.gemm(weight, 'N', gradOutput, 'N', 1, 0);
	return gradInput;
}

void
LinearSoftmax_AG::updateParameters(float learningRate) {
	gradWeight.gemm(input, 'N', gradOutput, 'T', 1, 0);
	gradBias.gemv(gradOutput, 'N', V1col, 1, 0);

	cumulGradWeight+=gradWeight.averageSquare();
	cumulGradBias+=gradBias.averageSquare();
	// for test
	/*cout << "LinearSoftmax_AG::updateParameters cumulGradWeight: " << cumulGradWeight << endl;
	cout << "LinearSoftmax_AG::updateParameters cumulGradBias: " << cumulGradBias << endl;*/

	weight.scal(1 - learningRate * weightDecay/sqrt(cumulGradWeight));
	weight.axpy(gradWeight, -learningRate/sqrt(cumulGradWeight));
	bias.axpy(gradBias, -learningRate/sqrt(cumulGradBias));
}

void
LinearSoftmax_AG::read(ioFile *iof) {
	iof->readString(name);
	// for test
	//cout << "LinearSoftmax_AG::read name: " << name << endl;
	weight.read(iof);
	bias.read(iof);
	iof->readFloat(cumulGradWeight);
	iof->readFloat(cumulGradBias);
}
void
LinearSoftmax_AG::write(ioFile * iof) {
	// for test
	//cout << "LinearSoftmax_AG::write for ag" << endl;
	iof->writeString(name);
	weight.write(iof);
	bias.write(iof);
	iof->writeFloat(cumulGradWeight);
	// for test
	//cout << "LinearSoftmax_AG::write cumulGradWeight: " << cumulGradWeight << endl;
	iof->writeFloat(cumulGradBias);
	// for test
	//cout << "LinearSoftmax_AG::write cumulGradBias: " << cumulGradBias << endl;
}
