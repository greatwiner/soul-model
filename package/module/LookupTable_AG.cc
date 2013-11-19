#include "mainModule.H"

LookupTable_AG::LookupTable_AG() {

}

LookupTable_AG::LookupTable_AG(int indexNumber, int dimensionSize, int inputSize,
    int blockSize, int oneClass, outils* otl) {
	name = "LookupTable_AG";
	weight.resize(dimensionSize, indexNumber);
	output.resize(dimensionSize * inputSize, blockSize);
	this->otl = otl;
	this->blockSize = blockSize;
	cumulGradWeight.resize(1, indexNumber);
	if (!oneClass) {
		reset();
    }
	else {
		init1class();
    }
	this->dimensionSize = dimensionSize;
	this->indexNumber = indexNumber;
}

LookupTable_AG::~LookupTable_AG() {
	// for test
	//cout << "LookupTable_AG::~LookupTable_AG here" << endl;
}

void
LookupTable_AG::reset() {
	weight.uniform(LKT_INIT0, LKT_INIT1, otl);
	cumulGradWeight=INIT_VALUE_ADAG;
}

void
LookupTable_AG::updateParameters(float learningRate) {
	int x0, x1;
	for (int i = 0; i < input.size[1]; i++) {
		x0 = 0;
		x1 = dimensionSize - 1;
		for (int j = 0; j < input.size[0]; j++) {
			selectWeight.select(weight, 1, input(j, i));
			selectGradWeight.sub(gradWeight, x0, x1, i, i);
			cumulGradWeight(input(j, i)) += selectGradWeight.averageSquare();
			if (weightDecay != 0) {
				// y = y - lr * wd * y
				selectWeight.scal(1 - learningRate * weightDecay/sqrt(cumulGradWeight(input(j, i))));
			}
			selectWeight.axpy(selectGradWeight, -learningRate/sqrt(cumulGradWeight(input(j, i))));
			x0 += dimensionSize;
			x1 += dimensionSize;
		}
	}
}

void
LookupTable_AG::read(ioFile* iof) {
	iof->readString(name);
	// for test
	//cout << "LookupTable_AG::read name: " << name << endl;
	weight.read(iof);
	cumulGradWeight.read(iof);
}

void
LookupTable_AG::write(ioFile* iof) {
	iof->writeString(name);
	weight.write(iof);
	cumulGradWeight.write(iof);
}
