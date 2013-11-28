#include "text.H"
NgramNCEWordTranslationDataSet::NgramNCEWordTranslationDataSet(int type, int n,
		int BOS, SoulVocab* inputVoc, SoulVocab* outputVoc, int mapIUnk,
		int mapOUnk, int maxNgramNumber) {
	this->type = type;
	this->n = n;
	// in translation model, the length of n-gram is 2*n for source part and for target part
	nm = n * 2;
	this->BOS = BOS;
	if (this->BOS > n) {
		this->BOS = this->n;
	}
	this->inputVoc = inputVoc;
	this->outputVoc = outputVoc;
	this->mapIUnk = mapIUnk;
	this->mapOUnk = mapOUnk;
	data = NULL;
	ngramNumber = 0;
	this->maxNgramNumber = maxNgramNumber;
	dataTensor.resize(1, 1);
	try {
		data = new int[maxNgramNumber * (nm + 3)];
		coef = new float[maxNgramNumber];
	} catch (bad_alloc& ba) {
		cerr << "bad_alloc caught: " << ba.what() << endl;
		exit(1);
	}
	sorted = 0;
}

NgramNCEWordTranslationDataSet::~NgramNCEWordTranslationDataSet() {
	if (coef != NULL) {
		delete[] coef;
	}
}

int NgramNCEWordTranslationDataSet::resamplingSentence(int totalLineNumber,
		int resamplingLineNumber, int* resamplingLineId) {
	if (totalLineNumber == resamplingLineNumber) {
		int i;
		for (i = 0; i < totalLineNumber; i++) {
			resamplingLineId[i] = i;
		}
		return 1;
	} else {
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
		sort(resamplingLineId, resamplingLineId + resamplingLineNumber);
		return 1;
	}
}

int NgramNCEWordTranslationDataSet::readText(ioFile* iof) {
	// Cannot read text for word align models
	if (type == 4 || type == 5) {
		cerr << "ERROR: dwt with type " << type
				<< " can not be used with text files" << endl;
		exit(1);
	}
	int readLineNumber = 0;
	// for test
	//cout << "NgramNCEWordTranslationDataSet::readText here" << endl;
	while (!iof->getEOF()) {
		// for test
		//cout << "NgramNCEWordTranslationDataSet::readText readLineNumber: "
				//<< readLineNumber << endl;
		addLine(iof);
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
	// for test
	//cout << "NgramNCEWordTranslationDataSet::readText readText dataSet: "
			//<< endl;
	//this->writeReBiNgram();
	return ngramNumber;
}

int NgramNCEWordTranslationDataSet::resamplingText(ioFile* iof,
		int totalLineNumber, int resamplingLineNumber) {
	// Cannot read text for word align models
	if (type == 4 || type == 5) {
		cerr << "ERROR: dwt with type " << type
				<< " can not be used with text file" << endl;
		exit(1);
	}
	int* resamplingLineId = new int[resamplingLineNumber];
	resamplingSentence(totalLineNumber, resamplingLineNumber, resamplingLineId);

	int readLineNumber = 0;
	int currentId = 0;
	string line;
	int numberOfReadLines = 0;
	while (!iof->getEOF()
			&& readLineNumber <= resamplingLineId[resamplingLineNumber - 1]) {
		if (readLineNumber != resamplingLineId[currentId]) {
			line = "";
			numberOfReadLines = 0;
			while (!this->eosLine(line)) {
				iof->getLine(line);
				numberOfReadLines += 1;
				if (numberOfReadLines > 1
						&& numberOfReadLines % MAX_WORD_PER_SENTENCE == 0) {
					cout
							<< "NgramNCEWordTranslationDataSet::resamplingText sentence is too long"
							<< endl;
				}
			}
		} else {
			currentId++;
			addLine(iof);
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
	delete[] resamplingLineId;
#if PRINT_DEBUG
	cout << endl;
#endif
	return ngramNumber;
}

intTensor&
NgramNCEWordTranslationDataSet::createTensor() {
	dataTensor.haveMemory = 0;
	dataTensor.size[0] = ngramNumber;
	dataTensor.size[1] = nm + 3;
	dataTensor.stride[0] = nm + 3;
	dataTensor.stride[1] = 1;
	if (dataTensor.data != data) {
		delete[] dataTensor.data;
		dataTensor.data = data;
	}
	if (groupContext) {
		sortNgram();
	}

	int ngramId;
	int preNgramId = 0;
	int i;
	int equal = 1;
	for (ngramId = 0; ngramId < ngramNumber - 1; ngramId++) {
		equal = 1;
		for (i = 0; i < nm - 1; i++) {

			if (data[ngramId * (nm + 3) + i]
					!= data[(ngramId + 1) * (nm + 3) + i]) {
				equal = 0;
				break;
			}
		}
		if (equal == 0 || !groupContext) {
			data[preNgramId * (nm + 3) + nm + 2] = ngramId + 1;
			preNgramId = ngramId + 1;
		}
	}
	if (equal == 1) {
		data[preNgramId * (nm + 3) + nm + 2] = ngramNumber;
	}
	data[ngramNumber * (nm + 3) - 1] = ngramNumber;
	probTensor.resize(ngramNumber, 1);
	return dataTensor;
}

int NgramNCEWordTranslationDataSet::addLine(string line) {
	cerr
			<< "ERROR: addLine(string line) is called with NgramNCEWordTranslationDataSet"
			<< endl;
	return 1;
}

int NgramNCEWordTranslationDataSet::addLine(ioFile* iof) {
	// for test
	//cout << "NgramNCEWordTranslationDataSet::addLine here" << endl;
	int i = 0;
	string line;
	string headline;
	headline = "";
	int currentId = 0;
	int inputIndex[MAX_WORD_PER_SENTENCE];
	int outputIndex[MAX_WORD_PER_SENTENCE];
	int unkIndex[MAX_WORD_PER_SENTENCE];
	int inputLength = 0;
	int outputLength = 0;
	string word;
	int inputCount;
	int outputCount;
	// memorize the ngramNumber before starting
	int oldNgramNumber = ngramNumber;
	// coefficient corresponding to the sentence
	float coefOfSentence;
	// PREFIX_SOURCE is normally 'src.' to distinguish between source words
	// and target words, for example comma: , and src.,
	string preSrc = PREFIX_SOURCE;
	if (iof->getLine(line)) {
		// for test
		//cout << "NgramNCEWordTranslationDataSet::addLine line: " << line << endl;
		// Is this test really necessary :S
		if (this->eosLine(line)) {
			return ngramNumber;
		}
		currentId++;
		// Initialize with SS token
		for (i = 0; i < BOS; i++) {
			inputIndex[inputLength] = inputVoc->ss;
			outputIndex[outputLength] = inputVoc->ss;
			unkIndex[outputLength] = outputVoc->ss;
			inputLength++;
			outputLength++;
		}

		do {
			// for test
			//cout << "NgramNCEWordTranslationDataSet::addLine line: " << line << endl;
			istringstream streamLine(line);
			streamLine >> word;
			inputCount = 0;
			// Read source words until meet separator |||
			while (word != "|||") {
				inputIndex[inputLength] = inputVoc->index(preSrc + word);
				if (mapIUnk && inputIndex[inputLength] == ID_UNK) {
					inputIndex[inputLength] = inputVoc->unk;
				}
				// For SrcTrg, Src models, they are also predicted words
				if (type == 2 || type == 3) {
					unkIndex[inputLength] = outputVoc->index(preSrc + word);
					if (mapOUnk && (unkIndex[inputLength] == ID_UNK)) {
						unkIndex[inputLength] = outputVoc->unk;
					}
				}
				inputLength++;
				inputCount++;
				streamLine >> word;
			}

			outputCount = 0;
			// Read target words
			while (streamLine >> word) {
				outputIndex[outputLength] = inputVoc->index(word);
				if (mapIUnk && (outputIndex[outputLength] == ID_UNK)) {
					outputIndex[outputLength] = inputVoc->unk;
				}
				// For Trg, TrgSrc models, they are also predicted words
				if (type == 0 || type == 1) {
					unkIndex[outputLength] = outputVoc->index(word);
					if (mapOUnk && (unkIndex[outputLength] == ID_UNK)) {
						unkIndex[outputLength] = outputVoc->unk;
					}
				}
				outputLength++;
				outputCount++;
			}
			// Depending on the type, add n-gram with appropriate position
			if (type == 0) {
				for (i = 0; i < outputCount; i++) {
					addDisWordTuple(inputIndex + inputLength - n,
							outputIndex + outputLength - outputCount + i + 1
									- n,
							unkIndex + outputLength - outputCount + i + 1 - n);
				}
			} else if (type == 2) {
				for (i = 0; i < inputCount; i++) {
					addDisWordTuple(outputIndex + outputLength - n,
							inputIndex + inputLength - inputCount + i + 1 - n,
							unkIndex + inputLength - inputCount + i + 1 - n);
				}
			} else if (type == 1) {
				for (i = 0; i < outputCount; i++) {
					addDisWordTuple(inputIndex + inputLength - n - inputCount,
							outputIndex + outputLength - outputCount + i + 1
									- n,
							unkIndex + outputLength - outputCount + i + 1 - n);
				}
			} else if (type == 3) {
				for (i = 0; i < inputCount; i++) {
					addDisWordTuple(
							outputIndex + outputLength - n - outputCount,
							inputIndex + inputLength - inputCount + i + 1 - n,
							unkIndex + inputLength - inputCount + i + 1 - n);
				}
			}
			iof->getLine(line);
			// for test
			//cout << "NgramNCEWordTranslationDataSet::addLine eosLine: " << this->eosLine(line) << endl;
		} while (!this->eosLine(line));
		// for test
		//cout << "NgramNCEWordTranslationDataSet::addLine here 1" << endl;
		// Finish one sentence with ES, adding n-gram ending with ES
		inputIndex[inputLength] = inputVoc->es;
		outputIndex[outputLength] = inputVoc->es;
		if (type == 0 || type == 1) {
			unkIndex[outputLength] = outputVoc->es;
			if (mapOUnk && (unkIndex[outputLength] == ID_UNK)) {
				unkIndex[outputLength] = outputVoc->unk;
			}
		} else if (type == 2 || type == 3) {
			unkIndex[inputLength] = outputVoc->es;
			if (mapOUnk && (unkIndex[inputLength] == ID_UNK)) {
				unkIndex[inputLength] = outputVoc->unk;
			}
		}
		inputLength++;
		outputLength++;
		// for test
		//cout << "NgramNCEWordTranslationDataSet::addLine here 2" << endl;
		if (type == 0) {
			addDisWordTuple(inputIndex + inputLength - n,
					outputIndex + outputLength - n,
					unkIndex + outputLength - n);
		} else if (type == 2) {
			addDisWordTuple(outputIndex + outputLength - n,
					inputIndex + inputLength - n, unkIndex + inputLength - n);
		} else if (type == 1) {
			addDisWordTuple(inputIndex + inputLength - n - 1,
					outputIndex + outputLength - n,
					unkIndex + outputLength - n);
		} else if (type == 3) {
			addDisWordTuple(outputIndex + outputLength - n - 1,
					inputIndex + inputLength - n, unkIndex + inputLength - n);
		}
		// now, read the coefficient and add to all examples which have just been added
		coefOfSentence = this->getCoefFromString(line);
		for (i = oldNgramNumber; i < ngramNumber; i++) {
			coef[i] = coefOfSentence;
		}
	}
	return ngramNumber;
}

int NgramNCEWordTranslationDataSet::readTextNgram(ioFile* iof) {
	cerr << "ERROR: readTextNgram is called with NgramNCEWordTranslationDataSet"
			<< endl;
	exit(1);
}

void NgramNCEWordTranslationDataSet::writeReBiNgram(ioFile* iof) {
	// for test
	//cout << "NgramNCEWordTranslationDataSet::writeReBiNgram here" << endl;
	//string* inputWordsByIndex = new string[inputVoc->wordNumber];
	//string* outputWordsByIndex = new string[outputVoc->wordNumber];
	//inputVoc->getWordByIndex(inputWordsByIndex);
	//outputVoc->getWordByIndex(outputWordsByIndex);
	int i = 0;
	iof->writeInt(this->realNgramNumberAfterGrouping);
	iof->writeInt(nm);
	int ngramId = 0;
	for (ngramId = 0; ngramId < ngramNumber; ngramId++) {
		// write only n-grams with non-zero coefficients
		if (coef[data[ngramId * (nm + 3) + nm + 1]] != 0) {
			iof->writeIntArray(data + ngramId * (nm + 3), nm);
			// for test
			//cout << "NgramNCEWordTranslationDataSet::writeReBiNgram write: " << endl;
			/*for (i = 0; i < nm - 1; i ++) {
				cout << inputWordsByIndex[data[ngramId * (nm + 3) + i]] << " ";
			}*/
			//cout << outputWordsByIndex[data[ngramId * (nm + 3) + nm - 1]] << " ";
			iof->writeFloat(coef[data[ngramId * (nm + 3) + nm + 1]]);
			// for test
			//cout << coef[data[ngramId * (nm + 3) + nm + 1]] << endl;
		}
	}
	// for test
	/*delete[] inputWordsByIndex;
	delete[] outputWordsByIndex;*/
}

int NgramNCEWordTranslationDataSet::readCoBiNgram(ioFile* iof) {
	int readLineNumber = 0;
	int i;
	int N;
	int ngramNumberInFile;
	iof->readInt(ngramNumberInFile);
	iof->readInt(N);
	int readTextNgram[N];
	int offset = N - nm;
	if (offset < 0) {
		cerr << "ERROR: order in id file is too small:" << N << " < " << nm
				<< endl;
		exit(1);
	}
	while (!iof->getEOF()) {
		iof->readIntArray(readTextNgram, N);
		// read coefficient for each ngram
		iof->readFloat(coef[ngramNumber]);
		if (iof->getEOF()) {
			break;
		}
		for (i = 0; i < nm; i++) {
			data[ngramNumber * (nm + 3) + i] = readTextNgram[offset + i];
		}
		data[ngramNumber * (nm + 3) + nm] = ID_END_NGRAM;
		data[ngramNumber * (nm + 3) + nm + 1] = ngramNumber;
		data[ngramNumber * (nm + 3) + nm + 2] = 0;
		ngramNumber++;
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

int NgramNCEWordTranslationDataSet::addDisWordTuple(int* srcIndex,
		int* desIndex, int* unkIndex) {
	int i;
	int use = 1;
	if (unkIndex[n - 1] == ID_UNK) {
		use = 0;
	} else {
		for (i = 0; i < n; i++) {
			if (srcIndex[i] == ID_UNK || desIndex[i] == ID_UNK) {
				use = 0;
				break;
			}
			data[ngramNumber * (nm + 3) + i] = srcIndex[i];
			data[ngramNumber * (nm + 3) + n + i] = desIndex[i];
		}
		if (use) {
			data[ngramNumber * (nm + 3) + nm - 1] = unkIndex[n - 1];
			data[ngramNumber * (nm + 3) + nm] = ID_END_NGRAM;
			//Normaly, is the order of this ngram in file
			data[ngramNumber * (nm + 3) + nm + 1] = ngramNumber;
			//The value to find the indentical context and ...
			data[ngramNumber * (nm + 3) + nm + 2] = 0;
			ngramNumber++;
		}
	}
	return ngramNumber;
}

void NgramNCEWordTranslationDataSet::shuffle(int times) {
	// for test
	cout << "NgramNCEWordTranslationDataSet::shuffle shuffle" << endl;
	// firstly, grouping n-grams
	this->realNgramNumberAfterGrouping = groupingNgram();
	int n3 = nm + 3;
	int *tg = new int[n3 * sizeof(int)];
	int i;
	int p1, p2;
	for (i = 0; i < times * ngramNumber; i++) {
		p1 = (int) (ngramNumber * drand48());
		p2 = (int) (ngramNumber * drand48());
		memcpy(tg, data + p1 * n3, n3 * sizeof(int));
		memcpy(data + p1 * n3, data + p2 * n3, n3 * sizeof(int));
		memcpy(data + p2 * n3, tg, n3 * sizeof(int));
	}
	sorted = 0;
}

int tupleNCECompare(const void *ngram1, const void *ngram2) {

	int i;
	int *pNgram1;
	int *pNgram2;

	pNgram1 = (int *) ngram1;
	pNgram2 = (int *) ngram2;
	i = 0;
	do {

		if (pNgram1[i] < pNgram2[i]) {
			return -1;
		} else {
			if (pNgram1[i] > pNgram2[i]) {
				return 1;
			}
		}
		i++;
	} while (pNgram1[i] != ID_END_NGRAM);
	return 0;

}

void NgramNCEWordTranslationDataSet::sortNgram() {
	// for test
	cout << "NgramNCEWordTranslationDataSet::sortNgram sort" << endl;
	if (sorted == 0) {
		qsort((void*) data, (size_t) ngramNumber,
				(nm + 3) * sizeof(unsigned int), tupleNCECompare);
	} else {
		cout << "NgramNCEWordTranslationDataSet::sortNgram data has been sorted"
				<< endl;
	}
	sorted = 1;
}

int NgramNCEWordTranslationDataSet::writeReBiNgram() {
	int i;
	int ngramId = 0;
	string* inputWordsByIndex = new string[inputVoc->wordNumber];
	string* outputWordsByIndex = new string[outputVoc->wordNumber];
	inputVoc->getWordByIndex(inputWordsByIndex);
	outputVoc->getWordByIndex(outputWordsByIndex);
	int realNgramNumber = 0;
	for (ngramId = 0; ngramId < ngramNumber; ngramId++) {
		// write only n-grams with non-zero coefficients
		if (coef[data[ngramId * (nm + 3) + nm + 1]] != 0) {
			for (i = 0; i < nm - 1; i++) {
				cout << inputWordsByIndex[data[ngramId * (nm + 3) + i]] << " ";
			}
			cout << outputWordsByIndex[data[ngramId * (nm + 3) + nm - 1]]
					<< " ";
			cout << data[ngramId * (nm + 3) + nm] << " ";
			cout << data[ngramId * (nm + 3) + nm + 1] << " ";
			cout << data[ngramId * (nm + 3) + nm + 2] << " ";
			cout << coef[data[ngramId * (nm + 3) + nm + 1]] << endl;
			realNgramNumber += 1;
		} else {
			cout
					<< "NgramNCEWordTranslationDataSet::writeReBiNgram coefficient 0: "
					<< endl;
			for (i = 0; i < nm - 1; i++) {
				cout << inputWordsByIndex[data[ngramId * (nm + 3) + i]] << " ";
			}
			cout << outputWordsByIndex[data[ngramId * (nm + 3) + nm - 1]]
					<< " ";
			cout << data[ngramId * (nm + 3) + nm] << " ";
			cout << data[ngramId * (nm + 3) + nm + 1] << " ";
			cout << data[ngramId * (nm + 3) + nm + 2] << " ";
			cout << coef[data[ngramId * (nm + 3) + nm + 1]] << endl;
		}
	}
	delete[] inputWordsByIndex;
	delete[] outputWordsByIndex;
	return realNgramNumber;
}

float NgramNCEWordTranslationDataSet::computePerplexity() {
	perplexity = 0;
	for (int i = 0; i < probTensor.length; i++) {
		// attention: coef and probTensor are in the same order (the order of n-gram before sortNgram() and shuffle())
		perplexity += -coef[i] * log(probTensor(i));
	}
	// for test
	cout << "NgramNCEWordTranslationDataSet::computePerplexity perplexity: "
			<< perplexity << endl;
	perplexity = perplexity / ngramNumber;
	return perplexity;
}

// grouping n-grams by summing coefficients
int NgramNCEWordTranslationDataSet::groupingNgram() {
	// rearrange the n-gram, sum coefficients over the same n-grams
	int realNgramNumber = 0;
	if (this->groupContext == 1) {
		// firstly, sort n-grams
		this->sortNgram();
		int ngramId = 0;
		int firstOccurOfNgram = ngramId;
		// sumOfCoef take the coef of firstOccurOfNgram
		float sumOfCoef = coef[data[firstOccurOfNgram * (nm + 3) + nm + 1]];
		int equal;
		int i;
		for (ngramId = 0; ngramId < ngramNumber - 1; ngramId++) {
			equal = 1;
			// compare n-grams, equal == 1 iff these are a same n-gram (context + word)
			for (i = 0; i < nm; i++) {
				if (data[ngramId * (nm + 3) + i]
						!= data[(ngramId + 1) * (nm + 3) + i]) {
					equal = 0;
					break;
				}
			}
			if (equal == 0) {
				coef[data[firstOccurOfNgram * (nm + 3) + nm + 1]] = sumOfCoef;
				if (sumOfCoef != 0) {
					// a real n-gram detected
					// we eliminate all n-grams with coefficient 0, so realNgramNumber ++ only when sumOfCoef != 0
					realNgramNumber += 1;
				}
				// re-initialize sumOfCoef and firstOccurOfNgram will be the next n-gram
				firstOccurOfNgram = ngramId + 1;
				sumOfCoef = coef[data[firstOccurOfNgram * (nm + 3) + nm + 1]];
			} else {
				sumOfCoef += coef[data[(ngramId + 1) * (nm + 3) + nm + 1]];
				coef[data[(ngramId + 1) * (nm + 3) + nm + 1]] = 0;
			}
		}
		if (equal == 1) {
			coef[data[firstOccurOfNgram * (nm + 3) + nm + 1]] = sumOfCoef;
		}
		if (sumOfCoef != 0) {
			// the last real n-gram has not been counted
			// we eliminate all n-grams with coefficient 0, so realNgramNumber ++ only when sumOfCoef != 0
			realNgramNumber += 1;
		}
	} else {
		realNgramNumber = ngramNumber;
	}
	return realNgramNumber;
}
