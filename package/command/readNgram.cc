#include "mainModel.H"

void
account(char* dataFileName, int blockSize, intTensor& contextOccurrences) {
	ioFile dataIof;
	dataIof.takeReadFile(dataFileName);
	int ngramNumber;
	dataIof.readInt(ngramNumber);
	cout << "readNgram::account ngramNumber: " << ngramNumber << endl;
	int N;
	dataIof.readInt(N);
	int blockNumber = ngramNumber / blockSize;
	int remainingNumber = ngramNumber - blockSize * blockNumber;
	int currentExampleNumber = 0;
	intTensor readTensor(blockSize, N);
	for (int i = 0; i < blockNumber; i++) {
		readTensor.readStrip(&dataIof);
		if (dataIof.getEOF()) {
			break;
		}
		currentExampleNumber += blockSize;
		for (int j = 0; j < blockSize; j++) {
			for (int k = 0; k < N-1; k++) {
				if (0 <= readTensor(j, k) < contextOccurrences.size[0]) {
					contextOccurrences(readTensor(j, k))+=1;
				}
			}
		}
	}
	if (remainingNumber != 0 && !dataIof.getEOF()) {
		intTensor lastReadTensor(remainingNumber, N);
		lastReadTensor.readStrip(&dataIof);
		for (int j = 0; j < remainingNumber; j++) {
			for (int k = 0; k < N-1; k++) {
				if (0 <= readTensor(j, k) < contextOccurrences.size[0]) {
					contextOccurrences(readTensor(j, k))+=1;
				}
			}
		}
	}
}

int
main(int argc, char *argv[]) {
	if (argc != 7) {
		cout << "dataFileNamePrefix blockSize maxVocSize minIte maxIte outFile" << endl;
		return 1;
	}
	char* dataFileNamePrefix = argv[1];
	int blockSize = atoi(argv[2]);
	int maxVocSize = atoi(argv[3]);
	intTensor contextOccurrences(maxVocSize, 1);
	int minIte = atoi(argv[4]);
	int maxIte = atoi(argv[5]);
	char* outFile = argv[6];
	char dataFileName[260];
	char convertStr[260];
	for (int ite = minIte; ite <= maxIte; ite++) {
		cout << "readNgram::main ite: " << ite << endl;
		strcpy(dataFileName, dataFileNamePrefix);
		sprintf(convertStr, "%ld", ite);
		strcat(dataFileName, convertStr);
		account(dataFileName, blockSize, contextOccurrences);
	}
	contextOccurrences.write();
	ioFile outWriteFile;
	outWriteFile.takeWriteFile(outFile);
	contextOccurrences.write(&outWriteFile);
	return 0;
}
