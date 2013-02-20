/*
 * prediction.cc
 *
 *  Created on: Dec 20, 2012
 *      Author: dokhanh
 */
#include "mainModel.H"
#include "ioFile.H"

float
prediction(NeuralModel* model, int start, int end, char* prefixModelFiles, char* dataFile, int n, int blockSize,
		string validType, int ngramNumber, int step, int allocation, int calDist, char* distFileName) {
	floatTensor probTensor;
	probTensor.resize(ngramNumber, 1);
	probTensor = 0;
	// for test
	//cout << "here i am" << endl;
	//Model* model = new NgramModel();
	int nIte = 0;
	ofstream distFile;
	distFile.open(distFileName, ios_base::app);
	for (int i = start; i <= end; i = i + step) {
		nIte += 1;
		// for test
		cout << "ite " << nIte << endl;
		char fileName[260];
		char cvstr[260];
		strcpy(fileName, prefixModelFiles);
		sprintf(cvstr, "%ld", i);
		strcat(fileName, cvstr);
		ioFile modelFile;
		modelFile.takeReadFile(fileName);
		floatTensor prevLookupTableRepre;
		if (calDist == 1 && i != start) {
			prevLookupTableRepre.copy(model->baseNetwork->lkt->weight);
		}
		if (allocation == 1 && i == start) {
			model->read(&modelFile, 1, blockSize);
			// for test
			model->computeProbability(model->dataSet, dataFile, validType);
		}
		else {
			model->read(&modelFile, 0, blockSize);
			model->computeProbability();
		}
		probTensor.axpy(model->dataSet->probTensor, 1);
		if (calDist == 1 && i != start) {
			prevLookupTableRepre.axpy(model->baseNetwork->lkt->weight, -1);
			cout << "Distance between iteration " << i << " and interation "<< i-step << " :" << endl;
			distFile << i-step << " " << prevLookupTableRepre.sumSquared() << endl;
			cout << prevLookupTableRepre.sumSquared() << endl;
		}
	}

	distFile.close();

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
	if (argc != 10) {
		cout << "start end prefixModelFiles dataFile n blockSize validType ngramNumber step" << endl;
		return 0;
	}
	int start = atoi(argv[1]);
	int end = atoi(argv[2]);
	char* prefixModelFiles = argv[3];
	char* dataFile = argv[4];
	int n = atoi(argv[5]);
	int blockSize = atoi(argv[6]);
	string validType = argv[7];
	int ngramNumber = atoi(argv[8]);
	int step = atoi(argv[9]);
	char outputPerSyn[260];
	strcpy(outputPerSyn, prefixModelFiles);
	strcat(outputPerSyn, "outper.Syn");
	ofstream outputPerpSyn;
	outputPerpSyn.open(outputPerSyn);
	int ind = start;
	NeuralModel* model = new NgramModel();
	char outputDistFileName[260];
	strcpy(outputDistFileName, prefixModelFiles);
	strcat(outputDistFileName, "out.squareDist");
	//ofstream outputDistFile;
	//outputDistFile.open(outputDistFileName);
	while (ind <= end) {
		// for test
		cout << "ind: " << ind << endl;
		int allo = 0;
		if (ind == start) {
			allo = 1;
		}
		int calDist = 0;
		if (ind == start) {
			calDist = 1;
		}
		float dist = 0.0;
		float per = prediction(model, ind, end, prefixModelFiles, dataFile, n, blockSize, validType,
				ngramNumber, step, allo, calDist, outputDistFileName);
		outputPerpSyn << ind << " " << per << endl;
		ind += step;
	}
	outputPerpSyn.close();
}
