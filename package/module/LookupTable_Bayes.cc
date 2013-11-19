#include "mainModule.H"

LookupTable_Bayes::LookupTable_Bayes(int indexNumber, int dimensionSize,
    int inputSize, int blockSize, int oneClass, outils* otl) : LookupTable(indexNumber,
    		dimensionSize, inputSize, blockSize, oneClass, otl)
{
  //prevWeight.resize(dimensionSize, indexNumber);
  //gradWeight.resize(dimensionSize, indexNumber);
  //prevGradWeight.resize(dimensionSize, indexNumber);
  // for lookuptable, output is the representations of the words in input
  // for test
  //cout << "here i am 5" << endl;
  // for test
  //cout << "here i am 5" << endl;
  // if oneClass == 0, we initialize the weights independently
  // if oneClass == 1, we initialize the weights so that all initial columns are the same
  pWeight.resize(weight);
  this->init1class();
}

LookupTable_Bayes::~LookupTable_Bayes() {

}

int
LookupTable_Bayes::numberOfWeights() {
	return indexNumber*output.size[0];
}

float
LookupTable_Bayes::sumSquaredWeights() {
	return this->weight.sumSquared();
}

void
LookupTable_Bayes::changeBlockSize(int blockSize)
{
  this->blockSize = blockSize;
  int size0 = output.size[0];
  output.resize(size0, blockSize);
  pWeight.resize(weight);
}

// if oneClass == 0
void
LookupTable_Bayes::reset()
{
        weight.uniform(LKT_INIT0, LKT_INIT1, otl);
        prevWeight.copy(weight);
        gradWeight = 0;
        prevGradWeight = 0;
}

// if oneClass == 1
void
LookupTable_Bayes::init1class()
{

}

floatTensor&
LookupTable_Bayes::backward(floatTensor& gradOutput, int last)
{
  /*int x0, x1;
  for (int i = 0; i < input.size[1]; i++) {

	  x0 = 0;
	  x1 = dimensionSize - 1;
	  for (int j = 0; j < input.size[0]; j++) {

		  // a portion of gradOutput
		  floatTensor selectGradOutput;
		  selectGradOutput.sub(gradOutput, x0, x1, i, i);
		  selectGradWeight.select(gradWeight, 1, input(j, i));
		  selectGradWeight.axpy(selectGradOutput, 1);
			  // a column of weight corresponding to the selectGradWeight being treated
		  selectWeight.select(weight, 1, input(j, i));
		  selectGradWeight.axpy(selectWeight, weightDecay);
			  // for test
			  //cout << "LookupTable_Bayes::backward weightDecay: " << weightDecay << endl;

		  x0 += dimensionSize;
		  x1 += dimensionSize;
	  }
  }*/
	gradWeight=gradOutput;
	/*if (gradWeight.testNan() != 0) {
		cout << "LookupTable_Bayes::backward gradWeight is nan" << endl;
	}*/

  return gradWeight;
}

void
LookupTable_Bayes::updateParameters(float learningRateForRd, float learningRateForParas, int last)
{
	//ioFile file;
	//file.takeWriteFile("/vol/work/dokhanh/wmt13/esLM/0/10gram/matrice");
	if (last==1) {
		weight.axpy(pWeight, sqrt(learningRateForParas));
	}
	else {
		/*if (gradWeight.testNan() != 0) {
			cout << "LookupTable_Bayes::updateParameters gradWeight is nan" << endl;
		}*/
		int x0, x1;
		for (int i = 0; i < input.size[1]; i++) {

			x0 = 0;
			x1 = dimensionSize - 1;
			for (int j = 0; j < input.size[0]; j++) {

				// a portion of gradOutput
				floatTensor selectGradOutput;
				selectGradOutput.sub(gradWeight, x0, x1, i, i);
				/*if (selectGradOutput.testNan() != 0) {
					cout << "LookupTable_Bayes::updateParameters selectGradOutput is nan" << endl;
					if (gradWeight.testNan() != 0) {
						cout << "LookupTable_Bayes::updateParameters because gradWeight is nan" << endl;
					}
					else {
						cout << "LookupTable_Bayes::updateParameters neu ko nan thi " << endl;
						//gradWeight.write();
						//cout << "LookupTable_Bayes::updateParameters testnan: " << gradWeight.testNanShow() << endl;
					}
				}*/
				selectWeight.select(weight, 1, input(j, i));
				selectWeight.scal(1 - sqrt(learningRateForRd*learningRateForParas)*weightDecay);
				selectWeight.axpy(selectGradOutput, -sqrt(learningRateForRd*learningRateForParas));

				x0 += dimensionSize;
				x1 += dimensionSize;
			}
		}
	}
	/*if (weight.testInf() != 0) {
		cout << "LookupTable_Bayes::updateParameters weight is inf, last=" << last << endl;
		//weight.copy(this->prevWeight);
		ioFile file;
		if (last==1) {
			file.takeWriteFile("/vol/work/dokhanh/wmt13/esLM/0/10gram/pWeight");
			pWeight.write(&file);
			cout << "LookupTable_Bayes::updateParameters pWeight: " << pWeight(89, 114814) << endl;
		}
		else {
			file.takeWriteFile("/vol/work/dokhanh/wmt13/esLM/0/10gram/gradWeight");
			gradWeight.write(&file);
		}
		exit(0);
	}*/
}

void
LookupTable_Bayes::updateRandomness(float learningRateForRd) {
	int x0, x1;
	for (int i = 0; i < input.size[1]; i++) {

		x0 = 0;
		x1 = dimensionSize - 1;
		for (int j = 0; j < input.size[0]; j++) {

			// a portion of gradOutput
			floatTensor selectGradOutput;
			floatTensor selectPWeight;
			selectGradOutput.sub(gradWeight, x0, x1, i, i);
			selectPWeight.select(pWeight, 1, input(j, i));
			selectPWeight.axpy(selectGradOutput, -sqrt(learningRateForRd));
			selectWeight.select(weight, 1, input(j, i));
			selectPWeight.axpy(selectWeight, -sqrt(learningRateForRd)*weightDecay);

			x0 += dimensionSize;
			x1 += dimensionSize;
		}
	}
	// for test
	cout << "LookupTable_Bayes::updateRandomness finish update" << endl;
}

void
LookupTable_Bayes::initializeP() {
	this->pWeight.initializeNormal(this->otl);
	cout << "LookupTable_Bayes::initializeP gaussian" << endl;
}

float
LookupTable_Bayes::getKinetic() {
	ki = 0.5*this->pWeight.sumSquared();
	return ki;
}

float
LookupTable_Bayes::getWeightDecayTerm() {
	wD = 0.5*weight.sumSquared();
	return wD;
}

float
LookupTable_Bayes::calculeH() {
	ki = this->getKinetic();
	wD = this->getWeightDecayTerm();
	// for test
	cout << "LookupTable_Bayes::calculeH kinetic: " << ki << endl;
	cout << "LookupTable_Bayes::calculeH weight decay: " << wD << endl;
	return ki + this->weightDecay*wD;
}

void
LookupTable_Bayes::reUpdateParameters(int accept) {
	if (accept == 1) {
		// we change the values
		prevWeight.copy(weight);
		prevGradWeight.copy(gradWeight);
	}
	else {
		// we keep the old values
		weight.copy(prevWeight);
		gradWeight.copy(prevGradWeight);
	}
}
