#include "mainModule.H"

LinearSoftmax_NCE::LinearSoftmax_NCE() {

}

LinearSoftmax_NCE::LinearSoftmax_NCE(int inputSize, int outputSize, int blockSize, outils* otl) {
	name = "LinearSoftmax_NCE";
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

	this->otl = otl;
	reset();
}

LinearSoftmax_NCE::~LinearSoftmax_NCE() {

}

void
LinearSoftmax_NCE::reset() {
	weight.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
	bias.uniform(LINEAR_INIT0, LINEAR_INIT1, otl);
}

floatTensor&
LinearSoftmax_NCE::backward(intTensor& word, floatTensor& coefTensor) {
	// gradOutput for weight and bias of the Linear part,
	// computed from the output after softmax (not the output of
	// the Linear part
	// coefTensor: this tensor is only for NCE algorithm
	// for test
	//cout << "LinearSoftmax::backward here" << endl;
	gradOutput.copy(output);
	// for test
	//cout << "LinearSoftmax::backward here1" << endl;
	int i;
	this->input = input;
	// for test
	//cout << "LinearSoftmax::backward here2" << endl;
	for (i = 0; i < blockSize; i++) {
		// If taking account this n-gram
		// In some cases, for some examples in the block, we don't
		// want to update with them, e.g., blockSize = 128 but in the
		// last block, we have only 78, predicted word for 50 *
		// last examples should be set to SIGN_NOT_WORD
		// If using, subtract its value in gradOutput 1
		selectGradOutput.select(gradOutput, 1, i);
		if (word(i) != SIGN_NOT_WORD) {
			gradOutput(word(i), i) -= 1;
			// multiply the gradient corresponding to word(i) with coefTensor(i)
			selectGradOutput.scal(coefTensor(i));
        }
		// Not use, all values = 0
		else {
			selectGradOutput = 0;
        }
    }
	// now we have calculated gradOutput, from which we calculate gradInput easily with the traditional chain rule
	gradInput.gemm(weight, 'N', gradOutput, 'N', 1, 0);
	return gradInput;
}

void
LinearSoftmax_NCE::updateParameters(float learningRate) {
	//As in Linear
	weight.gemm(input, 'N', gradOutput, 'T', -learningRate,
      1 - learningRate * weightDecay);
	bias.gemv(gradOutput, 'N', V1col, -learningRate, 1);
}

void
LinearSoftmax_NCE::read(ioFile* iof) {
	iof->readString(name);
	weight.read(iof);
	bias.read(iof);
}

void
LinearSoftmax_NCE::write(ioFile* iof) {
	iof->writeString(name);
	weight.write(iof);
	bias.write(iof);

	// not very beautiful here. When we take an ordinary model, and we transform it to a model trained using AdaGrad, the name is changed to sth_AG, but there have not been variables like cumulWeightGrad and cumulBiasGrad, so we need to write the value of these two variables to the file
	if (name == "LinearSoftmax_AG") {
		iof->writeFloat(INIT_VALUE_ADAG);
		iof->writeFloat(INIT_VALUE_ADAG);
	}
}
