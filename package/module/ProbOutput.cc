#include "mainModule.H"

ProbOutput::ProbOutput() {

}

ProbOutput::~ProbOutput() {

}

void
ProbOutput::changeBlockSize(int blockSize) {
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

floatTensor&
ProbOutput::forward(floatTensor& input) {
	this->input = input;
	// preOutput is the same as output in Linear.cc
	preOutput = 0;
	// preOutput = block_size bias columns
	preOutput.ger(bias, V1col, 1);

	// Then, preOutput = preOutput + weight^T x input
	preOutput.gemm(weight, 'T', input, 'N', 1, 1);

	// For each column, minus minimum value
	for (int i = 0; i < output.size[1]; i++) {
		// this value is chosen arbitrarily
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
	// output = e^(preOutput)
	output.mexp(preOutput);

	// softmaxVCol contains the sum for each column
	softmaxVCol.gemv(output, 'T', softmaxV1row, 1, 0);

	// For each column, divide by the sum to have
	// for each element e^x_i / \sum_j e^x_j
	for (int i = 0; i < output.size[1]; i++) {
		selectOutput.select(output, 1, i);
		selectOutput.scal(1.0 / softmaxVCol(i));
	}
	return output;
}

floatTensor&
ProbOutput::backward(floatTensor& word) {
	cerr << "ERROR: backward of ProbOutput for floatTensor" << endl;
	exit(1);
}

float
ProbOutput::distance2(Module& anotherOutput) {
	floatTensor distMatrix;
	distMatrix.copy(this->weight);
	distMatrix.axpy(anotherOutput.weight, -1);
	float res1 = distMatrix.sumSquared();
	distMatrix.resize(this->bias);
	distMatrix.copy(this->bias);
	distMatrix.axpy(anotherOutput.bias, -1);
	return res1+distMatrix.sumSquared();
}
