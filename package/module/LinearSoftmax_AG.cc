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
LinearSoftmax_AG::changeBlockSize(int blockSize) {
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
LinearSoftmax_AG::forward(floatTensor& input) {
	this->input = input;
	// preOutput is the same as output in Linear.cc
	preOutput = 0;
	// preOutput = block_size bias columns
	preOutput.ger(bias, V1col, 1);

	// Then, preOutput = preOutput + weight^T x input
	preOutput.gemm(weight, 'T', input, 'N', 1, 1);

	// For each column, minus minimum value
	for (int i = 0; i < output.size[1]; i++) {
		float max = -10000000;
		//float min = 10000000;
		for (int j = 0; j < output.size[0]; j++) {
			if (preOutput(j, i) > max) {
				max = preOutput(j, i);
			}
			/*if (preOutput(j ,i) < min) {
				min = preOutput(j, i);
			}*/
		}
		for (int j = 0; j < output.size[0]; j++) {
			preOutput(j, i) -= (max - 20);
		}
    }
	output.mexp(preOutput);

	// softmaxVCol contains the sum for each column
	softmaxVCol.gemv(output, 'T', softmaxV1row, 1, 0);

	// for test
	/*if (isinf(softmaxVCol.averageSquare())) {
		cout << "LinearSoftmax_AG::forward softmaxVcol is nan" << endl;
		softmaxVCol.write();
		exit(0);
	}*/

	// For each column, divide by the sum to have
	// for each element e^x_i / \sum_j e^x_j
	for (int i = 0; i < output.size[1]; i++) {
		selectOutput.select(output, 1, i);
		selectOutput.scal(1.0 / softmaxVCol(i));
		/*if (isnan(selectOutput.averageSquare())) {
			cout << "LinearSoftmax_AG::forward selectOutput is nan" << endl;
			selectOutput.write();
			cout << "LinearSoftmax_AG::forward softmaxVCol(i): " << softmaxVCol(i) << endl;
			exit(0);
		}*/
	}
	// for test
	/*float outputAS = output.averageSquare();
	if (isnan(outputAS)) {
		cout << "LinearSoftmax_AG::forward output is nan" << endl;
		output.write();
		exit(0);
	}*/
	return output;
}

floatTensor&
LinearSoftmax_AG::backward(floatTensor& word) {
	cerr << "ERROR: backward of LinearSoftmax for floatTensor" << endl;
	exit(1);
}

floatTensor&
LinearSoftmax_AG::backward(intTensor& word) {
	// gradOutput for weight and bias of the Linear part,
	// computed from the output after softmax (not the output of
	// the Linear part
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

float
LinearSoftmax_AG::distance2(LinearSoftmax& anotherOutput) {
	floatTensor distMatrix;
	distMatrix.copy(this->weight);
	distMatrix.axpy(anotherOutput.weight, -1);
	float res1 = distMatrix.sumSquared();
	distMatrix.resize(this->bias);
	distMatrix.copy(this->bias);
	distMatrix.axpy(anotherOutput.bias, -1);
	return res1+distMatrix.sumSquared();
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
