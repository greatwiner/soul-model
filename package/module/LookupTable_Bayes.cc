#include "mainModule.H"

LookupTable_Bayes::LookupTable_Bayes(int indexNumber, int dimensionSize,
    int inputSize, int blockSize, int oneClass, outils* otl) : LookupTable(indexNumber,
    		dimensionSize, inputSize, blockSize, oneClass, otl)
{
  prevWeight.resize(dimensionSize, indexNumber);
  gradWeight.resize(dimensionSize, indexNumber);
  prevGradWeight.resize(dimensionSize, indexNumber);
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
	LookupTable::init1class();
  prevWeight.copy(weight);
  gradWeight = 0;
  prevGradWeight = 0;
}

floatTensor&
LookupTable_Bayes::backward(floatTensor& gradOutput, int last)
{
	// for test
	//cout << "LookupTable_Bayes::backward" << endl;

  /*gradWeight = gradOutput;
  // for test
  cout << "backward of Lookup: " << endl;
  gradOutput.info();*/
	// integer values to indicate the beginning and the end of a block
  int x0, x1;
  for (int i = 0; i < input.size[1]; i++) {
	  //input.size[1] = blockSize
	  // we consider each element of dataset
      // for test
	  //cout << "i = " << i << endl;
	  x0 = 0;
	  x1 = dimensionSize - 1;
	  for (int j = 0; j < input.size[0]; j++) {
		  // for test
		  //cout << "j = " << j << endl;
		  //select the representation of the input word j

		  // a portion of gradOutput
		  floatTensor selectGradOutput;
		  selectGradOutput.sub(gradOutput, x0, x1, i, i);
		  selectGradWeight.select(gradWeight, 1, input(j, i));
		  selectGradWeight.axpy(selectGradOutput, 1);
		  //if (last == 1) {
			  // a column of weight corresponding to the selectGradWeight being treated
			  selectWeight.select(weight, 1, input(j, i));
			  selectGradWeight.axpy(selectWeight, weightDecay);
		  //}

		  x0 += dimensionSize;
		  x1 += dimensionSize;
	  }
  }
  /*// for test
  cout << "LookupTable_Bayes::backward difference in lkt:" << endl;
  gradWeight.scal(-0.00001);
  gradWeight.write();*/
  return gradWeight;
}

void
LookupTable_Bayes::updateParameters(float learningRate)
{
	weight.axpy(pWeight, sqrt(2*learningRate));
}

void
LookupTable_Bayes::updateRandomness(float learningRate) {
	pWeight.axpy(gradWeight, -sqrt(0.5*learningRate));
}

void
LookupTable_Bayes::initializeP() {
	this->pWeight.initializeNormal();
}

float
LookupTable_Bayes::calculeH() {
	float h = 0.5*this->pWeight.sumSquared() + 0.5*weightDecay*weight.sumSquared();
	// for test
	cout << "LookupTable_Bayes::calculeH h: " << h << endl;
	return h;
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
