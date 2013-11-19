#include "mainModule.H"

Sequential_Bayes::Sequential_Bayes(int maxSize) : Sequential(maxSize) {

}

Sequential_Bayes::~Sequential_Bayes() {

}

floatTensor&
Sequential_Bayes::backward(floatTensor& gradOutput, int last) {
	currentGradOutput = gradOutput;

	Module* currentModule = modules[size - 1];
	Module* previousModule;
	int i;
	for (i = size - 2; i > -1; i--)
	{
	  previousModule = modules[i];
	  if (Linear_Bayes* d1 = dynamic_cast<Linear_Bayes*>(currentModule)) {
		  currentGradOutput = d1->backward(currentGradOutput, last);
	  }
	  else {
		  currentGradOutput = currentModule->backward(currentGradOutput);
	  }
	  currentModule = previousModule;
	}
	if (Linear_Bayes* d2 = dynamic_cast<Linear_Bayes*>(currentModule)) {
		currentGradOutput = d2->backward(currentGradOutput, last);
	}
	else {
		currentGradOutput = currentModule->backward(currentGradOutput);
	}
	// Also backward with Lookup Table
	currentGradOutput = static_cast<LookupTable_Bayes*>(lkt)->backward(currentGradOutput, last);
	return currentGradOutput;
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
Sequential_Bayes::updateParameters(float learningRateForRd, float learningRateForParas, int last) {
	if (LookupTable_Bayes* dnmc = static_cast<LookupTable_Bayes*>(this->lkt)) {
		dnmc->updateParameters(learningRateForRd, learningRateForParas, last);
	}
	for (int i = 0; i < this->size; i ++) {
		if (Linear_Bayes* d1 = dynamic_cast<Linear_Bayes*>(modules[i])) {
			d1->updateParameters(learningRateForRd, learningRateForParas, last);
		}
	}
}
