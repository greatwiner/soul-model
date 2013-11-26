/*******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Class for recurrent data set.
 *******************************************************************/

#include "text.H"
RecurrentDataSet::RecurrentDataSet(int n, SoulVocab* inputVoc,
    SoulVocab* outputVoc, int cont, int blockSize, int maxNgramNumber) {
	this->n = n;
	this->BOS = -1;
	this->inputVoc = inputVoc;
	this->outputVoc = outputVoc;
	this->cont = cont;
	this->blockSize = blockSize;
	this->mapIUnk = 1;
	this->mapOUnk = 1;
	data = NULL;
	ngramNumber = 0;
	dataTensor.resize(1, 1);
	try {
		data = new int[maxNgramNumber * (n + 3)]; //Two ints code information, the last is ID_END_NGRAM
    }
	catch (bad_alloc& ba) {
		cerr << "RecurrentDataSet bad_alloc caught: " << ba.what() << endl;
		exit(1);
    }
}

/*int
RecurrentDataSet::addLine(string line) {
	int j;
	int inputIndex[MAX_WORD_PER_SENTENCE];
	int outputIndex[MAX_WORD_PER_SENTENCE];
	istringstream streamLine(line);
	string word;
	int i = 0;
	int length = 0;
	int use;
	while (streamLine >> word) {
		inputIndex[length] = inputVoc->index(word);
		outputIndex[length] = outputVoc->index(word);
		if (mapIUnk && inputIndex[length] == ID_UNK) {
			inputIndex[length] = inputVoc->unk;
        }
		if (mapOUnk && outputIndex[length] == ID_UNK) {
			outputIndex[length] = outputVoc->unk;
        }
		length++;
    }

	// The line have no ngram, don't do anything
	if (length == BOS - 1) {
		return 0;
    }
	if (ngramNumber >= 1 && cont) {
		// copy the context from the precedent n-grams
		for (j = 0; j < n - 1; j++) {
			data[ngramNumber * (n + 3) + j] = data[(ngramNumber - 1) * (n + 3) + j + 1];
        }
    }
	else {
		// this context is a special context indicating the re-initialization of contextFeature
		for (j = 0; j < n - 2; j++) {
			data[ngramNumber * (n + 3) + j] = inputVoc->ss;
        }
		data[ngramNumber * (n + 3) + n - 2] = inputVoc->es;
    }
	data[ngramNumber * (n + 3) + n - 1] = outputIndex[i];
	data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
	//Normaly, is the order of this ngram in file
	data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
	//The value to find the indentical context and ...
	data[ngramNumber * (n + 3) + n + 2] = 0;
	ngramNumber++;
	int lastId = n - 1;
	if (lastId > length) {
		lastId = length;
    }
	for (i = 1; i < lastId; i++) {
		for (j = 0; j < n - 1; j++){
			// copy context from precedent n-grams
			data[ngramNumber * (n + 3) + j] = data[(ngramNumber - 1) * (n + 3) + j + 1];
        }
		data[ngramNumber * (n + 3) + n - 1] = outputIndex[i];
		data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
		//Normaly, is the order of this ngram in file
		data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
		//The value to find the indentical context and ...
		data[ngramNumber * (n + 3) + n + 2] = 0;
		ngramNumber++;
    }

	for (i = 0; i <= length - n; i++) {
		use = 1;
		for (j = 0; j < n - 1; j++) {
			if (inputIndex[i + j] == ID_UNK) {
				use = 0;
				break;
            }
			data[ngramNumber * (n + 3) + j] = inputIndex[i + j];
        }
		if (outputIndex[i + n - 1] == ID_UNK) {
			use = 0;
        }
		else {
			data[ngramNumber * (n + 3) + n - 1] = outputIndex[i + n - 1];
        }
		if (use) {
			data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
			//Normaly, is the order of this ngram in file
			data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
			//The value to find the indentical context and ...
			data[ngramNumber * (n + 3) + n + 2] = 0;
			ngramNumber++;
        }
    }
	return 1;
}*/

int
RecurrentDataSet::addLine(string line) {
	int j;
	int inputIndex[MAX_WORD_PER_SENTENCE];
	int outputIndex[MAX_WORD_PER_SENTENCE];
	istringstream streamLine(line);
	string word;
	int i = 0;
	int length = 0;
	int use;
	while (streamLine >> word) {
		inputIndex[length] = inputVoc->index(word);
		outputIndex[length] = outputVoc->index(word);
		if (mapIUnk && inputIndex[length] == ID_UNK) {
			inputIndex[length] = inputVoc->unk;
        }
		if (mapOUnk && outputIndex[length] == ID_UNK) {
			outputIndex[length] = outputVoc->unk;
        }
		length++;
    }

	// The line have no ngram, don't do anything
	if (length == BOS - 1) {
		return 0;
    }

	// add the first n-gram of the sentence depending on cont and ngramNumber: if cont = 1 and ngramNumber >= 1, we have to continue to expand precedent n-grams. If cont = 0 (it means that at the beginning of each sentence, we have to re-initialize the contextFeature), or if cont = 1 but ngramNumber = 0 (so this is the first n-gram of paragraph), we need to add a special context which indicates that we re-initialize the contextFeature
	if (ngramNumber >= 1 && cont) {
		// copy the context from the precedent n-grams
		for (j = 0; j < n - 1; j++) {
			data[ngramNumber * (n + 3) + j] = data[(ngramNumber - 1) * (n + 3) + j + 1];
        }
    }
	else {
		// this context is a special context indicating the re-initialization of contextFeature
		for (j = 0; j < n - 2; j++) {
			data[ngramNumber * (n + 3) + j] = inputVoc->ss;
        }
		data[ngramNumber * (n + 3) + n - 2] = inputVoc->es;
    }
	// the first word of sentence
	data[ngramNumber * (n + 3) + n - 1] = outputIndex[0];
	data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
	//Normaly, is the order of this ngram in file
	data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
	//The value to find the indentical context and ...
	data[ngramNumber * (n + 3) + n + 2] = 0;
	ngramNumber++;

	for (i = 1; i < length; i++) {
		for (j = 0; j < n - 1; j++) {
			data[ngramNumber * (n + 3) + j] = data[(ngramNumber - 1) * (n + 3) + j + 1];
        }
		data[ngramNumber * (n + 3) + n - 1] = outputIndex[i];
		data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
		//Normaly, is the order of this ngram in file
		data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
		//The value to find the indentical context and ...
		data[ngramNumber * (n + 3) + n + 2] = 0;
		ngramNumber++;
    }
	return 1;
}

int
RecurrentDataSet::resamplingSentence(int totalLineNumber, int resamplingLineNumber, int* resamplingLineId) {
	if (totalLineNumber == resamplingLineNumber) {
		int i;
		for (i = 0; i < totalLineNumber; i++) {
			resamplingLineId[i] = i;
        }
		return 1;
    }
	else {
		if (cont) {
			int chosenPos;
			int i;
			chosenPos = rand() % totalLineNumber;
			resamplingLineId[0] = chosenPos;
			for (i = 1; i < resamplingLineNumber; i++) {
				resamplingLineId[i] = (resamplingLineId[0] + i) % totalLineNumber;
            }
        }
		else {
			int* buff = new int[totalLineNumber];
			int chosenPos;
			int i;
			for (i = 0; i < totalLineNumber; i++) {
				buff[i] = i;
            }
			int pos = totalLineNumber;
			for (i = 0; i < resamplingLineNumber; i++) {
				chosenPos = rand() % pos;
				resamplingLineId[i] = buff[chosenPos];
				buff[chosenPos] = buff[pos - 1];
				pos--;
            }
			delete[] buff;
        }
		sort(resamplingLineId, resamplingLineId + resamplingLineNumber);
		return 1;
    }
}

int
RecurrentDataSet::readText(ioFile* iof) {

	// need createTensor later
	this->alreadyCreateTensor = 0;
	string line;
	int readLineNumber = 0;
	int currentId = 0;
	while (!iof->getEOF()) {
		if (iof->getLine(line)) {
			if (!checkBlankString(line)) {
				line = line + " " + ES;
				addLine(line);
            }
			currentId++;
        }
		readLineNumber++;
#if PRINT_DEBUG
		if (readLineNumber % NLINEPRINT == 0) {
			cout << readLineNumber << " ... " << flush;
        }
#endif
    }
#if PRINT_DEBUG
	cout << endl;
#endif
	return 1;
}

int
RecurrentDataSet::resamplingText(ioFile* iof, int totalLineNumber, int resamplingLineNumber) {
	int* resamplingLineId = new int[resamplingLineNumber];
	resamplingSentence(totalLineNumber, resamplingLineNumber, resamplingLineId);

	this->alreadyCreateTensor = 0;
	string line;
	string headline;
	headline = "";
	int readLineNumber = 0;
	int currentId = 0;
	while (!iof->getEOF()) {
		if (iof->getLine(line)) {
			if (readLineNumber == resamplingLineId[currentId]) {
				if (!checkBlankString(line)) {
					line = headline + line + " " + ES;
					addLine(line);
                }
				currentId++;
            }
			if (currentId == resamplingLineNumber) {
				break;
            }
        }

		readLineNumber++;
#if PRINT_DEBUG
		if (readLineNumber % NLINEPRINT == 0) {
			cout << readLineNumber << " ... " << flush;
        }
#endif
    }
#if PRINT_DEBUG
	cout << endl;
#endif
	delete[] resamplingLineId;
	return ngramNumber;

}

intTensor&
RecurrentDataSet::createTensor() {
	if (alreadyCreateTensor == 1) {
		dataTensor.haveMemory = 0;
		dataTensor.size[0] = ngramNumber;
		dataTensor.size[1] = n + 3;
		dataTensor.stride[0] = n + 3;
		dataTensor.stride[1] = 1;
		// dataTensor is a pointer, doesn't have data
		if (dataTensor.data != data) {
			delete[] dataTensor.data;
			dataTensor.data = data;
		}
		probTensor.resize(ngramNumber, 1);
	}
	// for test
	//cout << "RecurrentDataSet::createTensor here" << endl;
	intTensor pos;
	pos.resize(blockSize, 1);
	intTensor dis;
	dis.resize(blockSize, 1);
	// for test
	//cout << "RecurrentDataSet::createTensor blockSize: " << blockSize << endl;
	// outMBlock: the maximal ngram number of each paragraph (we have blockSize paragraphs)
	int outMBlock = findPos(pos, dis) / (n + 3);
	// for test
	//cout << "RecurrentDataSet::createTensor here 4" << endl;
	// outNgramNumber: the maximal total ngram number
	int outNgramNumber = blockSize * outMBlock;
	int rNBlock;
	int rBlockSize;
	intTensor ssArray;
	ssArray.resize(n + 2, 1);
	// for test
	//cout << "RecurrentDataSet::createTensor here 5" << endl;
	ssArray = inputVoc->ss;
    // for test
	//cout << "RecurrentDataSet::createTensor here 6" << endl;
	ssArray(n - 1) = SIGN_NOT_WORD;
	ssArray(n) = ID_END_NGRAM;
	ssArray(n + 1) = SIGN_NOT_WORD;
	// for test
	//cout << "RecurrentDataSet::createTensor here 1" << endl;

	dataTensor.resize(outNgramNumber, n + 3);
	int rN;
	// process each sentence of paragraph
	for (rNBlock = 0; rNBlock < outMBlock; rNBlock++) {
		// process each paragraph
		for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++) {
			if (rNBlock * (n + 3) < dis(rBlockSize)) {
				for (rN = 0; rN < n + 2; rN++) {
					dataTensor(rBlockSize + rNBlock * blockSize, rN) = data[pos(rBlockSize) + rNBlock * (n + 3) + rN];
                }
            }
			else {
				for (rN = 0; rN < n + 2; rN++) {
					dataTensor(rBlockSize + rNBlock * blockSize, rN) = ssArray(rN);
                }
            }
			dataTensor(rBlockSize + rNBlock * blockSize, n + 2) = rBlockSize + rNBlock * blockSize + 1;
        }
    }
	probTensor.resize(ngramNumber, 1);
	alreadyCreateTensor = 1;
	return dataTensor;
}

int
RecurrentDataSet::readTextNgram(ioFile* iof) {
	string line;
	int readLineNumber = 0;
	while (!iof->getEOF()) {
		if (iof->getLine(line)) {
			if (!checkBlankString(line)) {
				addLine(line);
            }
        }
		readLineNumber++;
#if PRINT_DEBUG
		if (readLineNumber % NLINEPRINT == 0) {
			cout << readLineNumber << " ... " << flush;
        }
#endif
    }
#if PRINT_DEBUG
	cout << endl;
#endif
	return ngramNumber;
}

void
RecurrentDataSet::writeReBiNgram(ioFile* iof) {
	intTensor pos;
	pos.resize(blockSize, 1);
	intTensor dis;
	dis.resize(blockSize, 1);
	int outMBlock = findPos(pos, dis); // the maximum number of n-gram in each paragraph * (n + 3)
	iof->writeInt(blockSize * outMBlock / (n + 3));
	iof->writeInt(n);
	iof->writeInt(blockSize);
	int rNBlock = 0;
	int rBlockSize;
	intTensor ssArray;
	ssArray.resize(n, 1);
	ssArray = inputVoc->ss;
	ssArray(n - 1) = SIGN_NOT_WORD;
	//Write first 'ngram'
	for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++) {
		if (rNBlock < dis(rBlockSize)) {
			iof->writeIntArray(data + pos(rBlockSize), n);
        }
		else {
			iof->writeIntArray(ssArray.data, n);
        }
    }
	//Now write only word or </s>
	for (rNBlock = n + 3; rNBlock < outMBlock; rNBlock += n + 3) {
		for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++) {
			if (rNBlock < dis(rBlockSize)) {
				iof->writeIntArray(data + pos(rBlockSize) + rNBlock + n - 1, 1);
            }
			else {
				iof->writeIntArray(ssArray.data + n - 1, 1);
            }
        }
    }
}

int
RecurrentDataSet::findPos(intTensor& pos, intTensor& dis) {
	// pos is the first index position of each paragraph
	// dis is the last real index position of each paragraph
	pos(0) = 0;
	int rBlockSize;
	// dBlockSize is approximately the number of n-grams in each paragraph
	int dBlockSize = ngramNumber / blockSize;
	if (dBlockSize * blockSize < ngramNumber) {
		dBlockSize++;
    }
	int max = 0;
	// maxPos is the maximum of pos
	int maxPos = (ngramNumber - 1) * (n + 3);
	for (rBlockSize = 1; rBlockSize < blockSize; rBlockSize++) {
		pos(rBlockSize) = pos(rBlockSize - 1) + (dBlockSize - 1) * (n + 3);
		if (pos(rBlockSize) > maxPos) {
			pos(rBlockSize) = maxPos;
        }
		else {
			while (data[pos(rBlockSize) + n - 1] != outputVoc->es) {
				pos(rBlockSize) += n + 3;
            }
			pos(rBlockSize) += n + 3;
			// pos(rBlockSize found)
        }
		dis(rBlockSize - 1) = pos(rBlockSize) - pos(rBlockSize - 1);
		if (max < dis(rBlockSize - 1)) {
			max = dis(rBlockSize - 1);
        }
    }
	dis(blockSize - 1) = ngramNumber * (n + 3) - pos(blockSize - 1);
	if (max < dis(blockSize - 1)) {
		max = dis(blockSize - 1);
    }
	return max;
}

int
RecurrentDataSet::readCoBiNgram(ioFile* iof) {
	int readLineNumber = 0;
	int outMBlock;
	int rNBlock = 0;
	int i;
	iof->readInt(outMBlock);
	int N;
	iof->readInt(N);
	int offset = N - n;
	if (offset < 0) {
		cout << "RecurrentDataSet::readCoBiNgram order in id file is too small" << endl;
		exit(1);
	}
	int readTextNgram[N];
	iof->readInt(blockSize);
	outMBlock = outMBlock/blockSize;
	// read first n-gram
	int rBlockSize;

	// read the first n-grams
	for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++) {
		iof->readIntArray(readTextNgram, N);
		for (i = 0; i < n; i ++) {
			data[ngramNumber * (n + 3) + i] = readTextNgram[offset + i];
		}
		data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
		data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
		data[ngramNumber * (n + 3) + n + 2] = ngramNumber + 1;
		ngramNumber++;
		readLineNumber++;
	}

	// then read only the last word
	int readWord;
	for (rNBlock = 1; rNBlock < outMBlock; rNBlock++) {
		for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++) {
			iof->readInt(readWord);
			for (i = 0; i < n - 1; i++) {
				data[ngramNumber * (n + 3) + i] = data[(ngramNumber - blockSize) * (n + 3) + i + 1];
			}
			data[ngramNumber * (n + 3) + n - 1] = readWord;
			data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
			data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
			data[ngramNumber * (n + 3) + n + 2] = ngramNumber + 1;
			ngramNumber++;
			readLineNumber++;
		}
	}
#if PRINT_DEBUG
	if (readLineNumber % NLINEPRINT == 0) {
		cout << readLineNumber << " ... " << flush;
	}
#endif
#if PRINT_DEBUG
	cout << endl;
#endif
	return ngramNumber;
}

float
RecurrentDataSet::computePerplexity() {
	perplexity = 0;
	for (int i = 0; i < probTensor.length; i++) {
		perplexity += log(probTensor(i));
    }
	perplexity = exp(-perplexity / ngramNumber);
	return perplexity;
}

int
RecurrentDataSet::addLine(ioFile* iof) {
	string line;
	if (iof->getLine(line)) {
		if (!checkBlankString(line)) {
			line = line + " " + ES;
			return addLine(line);
        }
    }
	return 0;
}

