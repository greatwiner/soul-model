#include "mainModule.H"

Embeddings::Embeddings() {

}

Embeddings::~Embeddings() {

}

void
Embeddings::changeBlockSize(int blockSize) {
	this->blockSize = blockSize;
	int size0 = output.size[0];
	output.resize(size0, blockSize);
}

void
Embeddings::init1class() {
	floatTensor initRealTensor;
	initRealTensor.resize(weight.size[0], 1);
	initRealTensor.uniform(LKT_INIT0, LKT_INIT1, otl);
	floatTensor initSelectWeight;
	int i;
	for (i = 0; i < weight.size[1]; i++) {
		initSelectWeight.select(weight, 1, i);
		initSelectWeight.copy(initRealTensor);
	}
}

floatTensor&
Embeddings::forward(floatTensor& input)
{
	cout << "Wrong call, must call with input is intTensor" << endl;
	return input;
}

floatTensor&
Embeddings::forward(intTensor& input) {
	this->input = input;
	int x0, x1;
	for (int i = 0; i < input.size[1]; i++) {
		x0 = 0;
		x1 = dimensionSize - 1;
		for (int j = 0; j < input.size[0]; j++) {
			selectOutput.sub(output, x0, x1, i, i);
			selectWeight.select(weight, 1, input(j, i));
			selectOutput.copy(selectWeight, weight);
			x0 += dimensionSize;
			x1 += dimensionSize;
        }
    }
	return output;
}

floatTensor&
Embeddings::backward(floatTensor& gradOutput) {
	gradWeight = gradOutput;
	// We do not use return variable later, so whatever you want :S
	return gradWeight;
}

float
Embeddings::distance2(Module& anotherLkt) {
	floatTensor distanceMatrix;
	distanceMatrix.copy(this->weight);
	distanceMatrix.axpy(anotherLkt.weight, -1);
	return distanceMatrix.sumSquared();
}

