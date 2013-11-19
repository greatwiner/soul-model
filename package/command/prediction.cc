/*
 * prediction.cc
 *
 *  Created on: Dec 20, 2012
 *      Author: dokhanh
 */
#include "mainModel.H"
#include "ioFile.H"

float
prediction(NeuralModel* model, char* modelFileName, char* dataFile, int n, int blockSize,
		string validType, int calDist) {
	floatTensor probTensor;
	int ngramNumber = 0;
	// for test
	//cout << "here i am" << endl;
	//Model* model = new NgramModel();
	int nIte = 0;
	ioFile modelFiles;
	modelFiles.takeReadFile(modelFileName);
	while (!modelFiles.getEOF()) {
		nIte += 1;
		// for test
		cout << "ite " << nIte << endl;
		string filename;
		cout << "prediction::prediction here" << endl;
		modelFiles.getLine(filename);
		cout << "prediction::prediction here 1" << endl;
		ioFile modelFile;
		char filename1[260];
		strcpy(filename1, filename.c_str());
		cout << "prediction::prediction here 3" << endl;
		// for test
		cout << "prediction::prediction filename1: " << filename1 << endl;
		int open=modelFile.takeReadFile(filename1);
		cout << "prediction::prediction here 2" << endl;
		if (open==0) {
			nIte -= 1;
			break;
		}
		cout << "Read file: " << filename << endl;
		floatTensor prevLookupTableRepre;
		if (calDist == 1 && nIte > 1) {
			prevLookupTableRepre.copy(model->baseNetwork->lkt->weight);
		}
		if (nIte == 1) {
			cout << "Allocation = 1" << endl;
			cout << "prediction::prediction here 4" << endl;
			model->read(&modelFile, 1, blockSize);
			cout << "prediction::prediction here 5" << endl;
			model->computeProbability(model->dataSet, dataFile, validType);
			cout << "prediction::prediction here 6" << endl;
		}
		else {
			cout << "Allocation = 0" << endl;
			if (PREDICTION_ALLO == 1) {
				model->read(&modelFile, 1, blockSize);
				model->computeProbability(model->dataSet, dataFile, validType);
			}
			else {
				model->read(&modelFile, 0, blockSize);
				model->computeProbability();
			}
		}
		if (nIte == 1) {
			ngramNumber = model->dataSet->ngramNumber;
			probTensor.resize(ngramNumber, 1);
			probTensor = 0;
		}
		probTensor.axpy(model->dataSet->probTensor, 1);
		if (calDist == 1 && nIte > 1) {
			floatTensor curLookupTableRepre;
			curLookupTableRepre.copy(model->baseNetwork->lkt->weight);
			floatTensor distLkt;
			distLkt.copy(curLookupTableRepre);
			distLkt.axpy(prevLookupTableRepre, -1);
			float distAngle = prevLookupTableRepre.angleDist(curLookupTableRepre);
			float distAbsAvg = distLkt.averageSquareBig();
			cout << "Squared average of current vector: ";
			cout << curLookupTableRepre.averageSquareBig() << endl;
			cout << "Average absolute distance between iteration " << nIte << " and iteration" << nIte-1 << " : ";
			cout << distAbsAvg << endl;
			cout << "Angle distance between iteration " << nIte << " and interation "<< nIte-1 << " : ";
			cout << distAngle << endl;
		}
	}
	probTensor.scal((float)1/nIte);
	float perplexity = 0;
	for (int i = 0; i < probTensor.length; i++) {
		perplexity += log(probTensor(i));
	}
	perplexity = exp(-perplexity / ngramNumber);
	return perplexity;
}

int
main(int argc, char *argv[]) {
	if (argc != 6) {
		cout << "modelFileName dataFile n blockSize validType" << endl;
		return 0;
	}
	char* modelFileName = argv[1];
	char* dataFile = argv[2];
	int n = atoi(argv[3]);
	int blockSize = atoi(argv[4]);
	string validType = argv[5];
	NeuralModel* model = new NgramModel();
	int allo = 1;
	int calDist = 1;
	float per = prediction(model, modelFileName, dataFile, n, blockSize, validType, calDist);
	cout << per << endl;
	delete model;
	return 0;
}
