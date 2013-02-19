#include "mainModule.H"

Sequential_Bayes::Sequential_Bayes(int maxSize) : Sequential(maxSize) {

}

Sequential_Bayes::~Sequential_Bayes() {

}

/*floatTensor&
Sequential_Bayes::backward(floatTensor& gradOutput)
{
  // gradOutput
  currentGradOutput = gradOutput;

  // pointers to modules of the base network
  Module* currentModule = modules[size - 1];
  Module* previousModule;
  int i;
  for (i = size - 2; i > -1; i--)
    {
      previousModule = modules[i];
      currentGradOutput = currentModule->backward(currentGradOutput);
      currentModule = previousModule;
    }
  // after the for iterations, currentModule = modules[0]
  currentGradOutput = currentModule->backward(currentGradOutput);
  currentGradOutput = lkt->backward(currentGradOutput);
  return currentGradOutput;// = gradWeight
}*/

floatTensor&
Sequential_Bayes::backward(floatTensor& gradOutput) {
	return Sequential::backward(gradOutput);
}

void
Sequential_Bayes::updateRandomness(float learningRate) {
	for (int i = 0; i < size; i ++) {
		if (Linear_Bayes* d1 = dynamic_cast<Linear_Bayes*>(modules[i])) {
			d1->updateRandomness(learningRate);
		}
	}
	if (LookupTable_Bayes* d2 = static_cast<LookupTable_Bayes*>(lkt)) {
		d2->updateRandomness(learningRate);
	}
}

int
Sequential_Bayes::numberOfWeights() {
	int num = 0;
	if (LookupTable_Bayes* d = static_cast<LookupTable_Bayes*>(this->lkt)) {
		num += d->numberOfWeights();
	}
	// for test
	//cout << "number of lkt: " << this->lkt->numberOfWeights() << endl;
	for (int i = 0;i < size;i++) {
		if (Linear_Bayes* d1 = static_cast<Linear_Bayes*>(modules[i])) {
			num += d1->numberOfWeights();
		}
	}
	return num;
}

float
Sequential_Bayes::sumSquaredWeights() {
	float sum = 0;
	if (LookupTable_Bayes* dnmc = static_cast<LookupTable_Bayes*>(this->lkt)) {
		sum += dnmc->sumSquaredWeights();
	}
	for (int i = 0;i < size;i++) {
		if (Linear_Bayes* d1 = static_cast<Linear_Bayes*>(modules[i])) {
			sum += d1->sumSquaredWeights();
		}
	}
	return sum;
}

void
Sequential_Bayes::initializeP() {
	if (LookupTable_Bayes* dnmc = static_cast<LookupTable_Bayes*>(this->lkt)) {
		dnmc->initializeP();
	}
	for (int i = 0; i < size; i ++) {
		if (Linear_Bayes* d1 = dynamic_cast<Linear_Bayes*>(modules[i])) {
			d1->initializeP();
		}
	}
}

float
Sequential_Bayes::calculeH() {
	float h;
	if (LookupTable_Bayes* dnmc = static_cast<LookupTable_Bayes*>(this->lkt)) {
		h = dnmc->calculeH();
	}
	for (int i = 0; i < this->size; i ++) {
		if (Linear_Bayes* d1 = dynamic_cast<Linear_Bayes*>(modules[i])) {
			h+=d1->calculeH();
		}
	}
	return h;
}

void
Sequential_Bayes::reUpdateParameters(int accept) {
	// for test
	//cout << "vi 1" << endl;
	if (LookupTable_Bayes* dnmc = static_cast<LookupTable_Bayes*>(this->lkt)) {
		dnmc->reUpdateParameters(accept);
	}
	// for test
	//cout << "vi 2" << endl;
	for (int i = 0; i < this->size; i ++) {
		// for test
		//cout << "vong thu " << i << endl;
		if (Linear_Bayes* d1 = dynamic_cast<Linear_Bayes*>(modules[i])) {
			d1->reUpdateParameters(accept);
		}
	}
}
