#include "text.H"

NgramNCEDataSet::NgramNCEDataSet(int type, int n, int BOS, SoulVocab* inputVoc,
		SoulVocab* outputVoc, int mapIUnk, int mapOUnk, int maxNgramNumber) {
	srand48 (time(NULL));srand(time(NULL));
	this->type = type;
	this->n = n;

	// instead of having n + 3 numbers for each ngram, we have n + 4
	// the additional one indicates its coefficient
	this->lengthPerNgram = n + 3;
	this->groupContext = 1;
	this->BOS = BOS;
	if (this->BOS > n - 1) {
		this->BOS = this->n - 1;
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
		data = new int[maxNgramNumber * this->lengthPerNgram];
		coef = new float[maxNgramNumber];
	}
	catch (bad_alloc& ba) {
		cerr << "bad_alloc caught: " << ba.what() << endl;
		exit(1);
	}
	sorted = 0;
}

NgramNCEDataSet::~NgramNCEDataSet() {
	if (coef != NULL) {
		delete[] coef;
	}
}

int NgramNCEDataSet::addLineWithCoef(string line, float coefficient) {
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
		// Line is too long
		if (length > MAX_WORD_PER_SENTENCE) {
			return 0;
		}
	}
	// Line has no ngram, don't do anything
	if (length == BOS - 1) {
		return 0;
	}
	for (i = 0; i <= length - n; i++) {
		use = 1;
		if (type == 0 || type == 1) {
			for (j = 0; j < n - 1; j++) {
				if (inputIndex[i + j] < 0) {
					use = 0;
					break;
				}
				data[ngramNumber * lengthPerNgram + j] = inputIndex[i + j];
			}
			if (outputIndex[i + n - 1] < 0) {
				use = 0;
			} else {
				data[ngramNumber * lengthPerNgram + n - 1] = outputIndex[i + n
						- 1];
			}
		} else {
			for (j = 0; j < n; j++) {
				if (inputIndex[i + j] < 0 && j != (n - 1) / 2) {
					use = 0;
					break;
				}
				if (j < (n - 1) / 2) {
					data[ngramNumber * lengthPerNgram + j] = inputIndex[i + j];
				} else if (j > (n - 1) / 2) {
					data[ngramNumber * lengthPerNgram + j - 1] = inputIndex[i
							+ j];
				}
			}
			if (outputIndex[i + (n - 1) / 2] < 0) {
				use = 0;
			} else {
				data[ngramNumber * lengthPerNgram + n - 1] = outputIndex[i
						+ (n - 1) / 2];
			}
		}
		if (use) {
			data[ngramNumber * lengthPerNgram + n] = ID_END_NGRAM;
			data[ngramNumber * lengthPerNgram + n + 1] = ngramNumber;
			data[ngramNumber * lengthPerNgram + n + 2] = 0;

			// coefficient indicating if the added ngrams are positive or negative, or how much they are positive or negative
			coef[ngramNumber] = coefficient;
			ngramNumber++;
			//Add negative example, copy context and random uniform predicted word
			/*this->setCoefficient(-1);
			 for (j = 0; j < n - 1; j++)
			 {
			 data[ngramNumber * (n + 3) + j] = data[(ngramNumber - 1)
			 * (n + 3) + j];
			 }
			 do
			 {
			 data[ngramNumber * (n + 3) + n - 1]
			 = (int) (outputVoc->wordNumber * drand48());
			 }
			 while (data[ngramNumber * (n + 3) + n - 1] == data[(ngramNumber - 1)
			 * (n + 3) + n - 1]);
			 data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
			 data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
			 data[ngramNumber * (n + 3) + n + 2] = 0;
			 ngramNumber++;*/
		}
	}
	return 1;
}

int NgramNCEDataSet::addLine(string line) {
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
		// Line is too long
		if (length > MAX_WORD_PER_SENTENCE) {
			return 0;
		}
	}
	// Line has no ngram, don't do anything
	if (length == BOS - 1) {
		return 0;
	}
	for (i = 0; i <= length - n; i++) {
		use = 1;
		if (type == 0 || type == 1) {
			for (j = 0; j < n - 1; j++) {
				if (inputIndex[i + j] < 0) {
					use = 0;
					break;
				}
				data[ngramNumber * lengthPerNgram + j] = inputIndex[i + j];
			}
			if (outputIndex[i + n - 1] < 0) {
				use = 0;
			} else {
				data[ngramNumber * lengthPerNgram + n - 1] = outputIndex[i + n
						- 1];
			}
		} else {
			for (j = 0; j < n; j++) {
				if (inputIndex[i + j] < 0 && j != (n - 1) / 2) {
					use = 0;
					break;
				}
				if (j < (n - 1) / 2) {
					data[ngramNumber * lengthPerNgram + j] = inputIndex[i + j];
				} else if (j > (n - 1) / 2) {
					data[ngramNumber * lengthPerNgram + j - 1] = inputIndex[i
							+ j];
				}
			}
			if (outputIndex[i + (n - 1) / 2] < 0) {
				use = 0;
			} else {
				data[ngramNumber * lengthPerNgram + n - 1] = outputIndex[i
						+ (n - 1) / 2];
			}
		}
		if (use) {
			data[ngramNumber * lengthPerNgram + n] = ID_END_NGRAM;
			data[ngramNumber * lengthPerNgram + n + 1] = ngramNumber;
			data[ngramNumber * lengthPerNgram + n + 2] = 0;

			// positive example
			coef[ngramNumber] = 1;
			ngramNumber++;
			//Add negative example, copy context and random uniform predicted word
			for (j = 0; j < n - 1; j++) {
				data[ngramNumber * lengthPerNgram + j] = data[(ngramNumber - 1)
						* lengthPerNgram + j];
			}
			do {
				data[ngramNumber * lengthPerNgram + n - 1] =
						(int) (outputVoc->wordNumber * drand48());
			} while (data[ngramNumber * lengthPerNgram + n - 1]
					== data[(ngramNumber - 1) * lengthPerNgram + n - 1]);
			data[ngramNumber * lengthPerNgram + n] = ID_END_NGRAM;
			data[ngramNumber * lengthPerNgram + n + 1] = ngramNumber;
			data[ngramNumber * lengthPerNgram + n + 2] = 0;

			// negative example
			coef[ngramNumber] = -1;
			ngramNumber++;
		}
	}
	return 1;
}

int NgramNCEDataSet::resamplingSentence(int totalLineNumber,
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

int NgramNCEDataSet::readTextNoCoef(ioFile* iof) {
	int i = 0;
	string line;
	string headline;
	string invLine;
	string tailline;
	headline = "";
	tailline = "";
	// Normal
	if (type == 0) {
		for (i = 0; i < BOS; i++) {
			headline = headline + SS + " ";
		}
		tailline = tailline + " " + ES;
	}
	// Inverse
	else if (type == 1) {
		for (i = 0; i < BOS; i++) {
			tailline = tailline + " " + ES;
		}
		headline = headline + SS + " ";
	}
	// Center
	else if (type == 2) {
		for (i = 0; i < BOS / 2; i++) {
			tailline = tailline + " " + ES;
			headline = headline + SS + " ";
		}
	}
	int readLineNumber = 0;
	while (!iof->getEOF()) {
		if (iof->getLine(line)) {
			if (!checkBlankString(line)) {
				line = headline + line + tailline;
				if (type == 0 || type == 2) {
					addLine(line);
				} else if (type == 1) {
					invLine = inverse(line);
					addLine(invLine);
				}
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
	return 1;
}

int NgramNCEDataSet::readText(ioFile* iof) {
	int i = 0;
	string line;
	string headline;
	string invLine;
	string tailline;
	headline = "";
	tailline = "";
	// Normal
	if (type == 0) {
		for (i = 0; i < BOS; i++) {
			headline = headline + SS + " ";
		}
		tailline = tailline + " " + ES;
	}
	// Inverse
	else if (type == 1) {
		for (i = 0; i < BOS; i++) {
			tailline = tailline + " " + ES;
		}
		headline = headline + SS + " ";
	}
	// Center
	else if (type == 2) {
		for (i = 0; i < BOS / 2; i++) {
			tailline = tailline + " " + ES;
			headline = headline + SS + " ";
		}
	}
	int readLineNumber = 0;
	float currentCoef;
	while (!iof->getEOF()) {
		if (iof->getLine(line)) {
			if (!checkBlankString(line)) {
				currentCoef = getCoefFromString(line);
				// line has been modified
				if (!checkBlankString(line)) {
					line = headline + line + tailline;
					if (type == 0 || type == 2) {
						addLineWithCoef(line, currentCoef);
					} else if (type == 1) {
						invLine = inverse(line);
						addLineWithCoef(invLine, currentCoef);
					}
				}
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
	// for test
	/*cout << "NgramNCEDataSet::readText data: " << endl;
	string* inputWordsByIndex = new string[inputVoc->wordNumber];
	string* outputWordsByIndex = new string[outputVoc->wordNumber];
	inputVoc->getWordByIndex(inputWordsByIndex);
	outputVoc->getWordByIndex(outputWordsByIndex);
	for (int i = 0; i < ngramNumber; i++) {
		for (int j = 0; j < n - 1; j++) {
			cout << inputWordsByIndex[data[i * (n + 3) + j]] << " ";
		}
		cout << outputWordsByIndex[data[i * (n + 3) + n - 1]] << " ";
		cout << coef[i];
		cout << endl;
	}
	delete[] inputWordsByIndex;
	delete[] outputWordsByIndex;*/
	return 1;
}

int NgramNCEDataSet::resamplingTextNoCoef(ioFile* iof, int totalLineNumber,
		int resamplingLineNumber) {
	int* resamplingLineId = new int[resamplingLineNumber];
	resamplingSentence(totalLineNumber, resamplingLineNumber, resamplingLineId);

	int i = 0;
	string line;
	string headline;
	string invLine;
	string tailline;
	headline = "";
	tailline = "";
	// Normal
	if (type == 0) {
		for (i = 0; i < BOS; i++) {
			headline = headline + SS + " ";
		}
		tailline = tailline + " " + ES;
	}
	// Inverse
	else if (type == 1) {
		for (i = 0; i < BOS; i++) {
			tailline = tailline + " " + ES;
		}
		headline = headline + SS + " ";
	}
	// Center
	else if (type == 2) {
		for (i = 0; i < BOS / 2; i++) {
			tailline = tailline + " " + ES;
			headline = headline + SS + " ";
		}
	}
	int readLineNumber = 0;
	int currentId = 0;
	while (!iof->getEOF()) {
		if (iof->getLine(line)) {
			// for test
			//cout << "NgramRankDataSet::resamplingText line: " << line << endl;
			if (readLineNumber == resamplingLineId[currentId]) {
				if (!checkBlankString(line)) {
					line = headline + line + tailline;
					if (type == 0 || type == 2) {
						addLine(line);
					} else if (type == 1) {
						invLine = inverse(line);
						addLine(invLine);
					}
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

int NgramNCEDataSet::resamplingText(ioFile* iof, int totalLineNumber,
		int resamplingLineNumber) {
	int* resamplingLineId = new int[resamplingLineNumber];
	resamplingSentence(totalLineNumber, resamplingLineNumber, resamplingLineId);

	int i = 0;
	string line;
	string headline;
	string invLine;
	string tailline;
	headline = "";
	tailline = "";
	// Normal
	if (type == 0) {
		for (i = 0; i < BOS; i++) {
			headline = headline + SS + " ";
		}
		tailline = tailline + " " + ES;
	}
	// Inverse
	else if (type == 1) {
		for (i = 0; i < BOS; i++) {
			tailline = tailline + " " + ES;
		}
		headline = headline + SS + " ";
	}
	// Center
	else if (type == 2) {
		for (i = 0; i < BOS / 2; i++) {
			tailline = tailline + " " + ES;
			headline = headline + SS + " ";
		}
	}
	int readLineNumber = 0;
	int currentId = 0;
	float currentCoef;
	while (!iof->getEOF()) {
		if (iof->getLine(line)) {
			// for test
			//cout << "NgramRankDataSet::resamplingText line: " << line << endl;
			if (readLineNumber == resamplingLineId[currentId]) {
				if (!checkBlankString(line)) {
					currentCoef = getCoefFromString(line);
					// line has been modified
					if (!checkBlankString(line)) {
						line = headline + line + tailline;
						if (type == 0 || type == 2) {
							addLineWithCoef(line, currentCoef);
						} else if (type == 1) {
							invLine = inverse(line);
							addLineWithCoef(invLine, currentCoef);
						}
					}
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
NgramNCEDataSet::createTensor() {
	dataTensor.haveMemory = 0;
	dataTensor.size[0] = ngramNumber;
	dataTensor.size[1] = lengthPerNgram;
	dataTensor.stride[0] = lengthPerNgram;
	dataTensor.stride[1] = 1;
	// dataTensor is a pointer, doesn't have data
	if (dataTensor.data != data) {
		delete[] dataTensor.data;
		dataTensor.data = data;
	}
	// Sort using quicksort
	// Because each ngram corresponds to a coefficient at position n + 3, we do not need to process in a fix order
	if (groupContext) {
		sortNgram();
	}
	// for test
	//cout << "NgramDataSet::createTensor dataTensor after sorting: " << endl;
	//dataTensor.write();
	// Edit info integer to keep the info for the first next n-gram
	// which has a different context
	int ngramId;
	int preNgramId = 0;
	int i;
	int equal = 1;
	for (ngramId = 0; ngramId < ngramNumber - 1; ngramId++) {
		equal = 1;
		for (i = 0; i < n - 1; i++) {

			if (data[ngramId * lengthPerNgram + i]
					!= data[(ngramId + 1) * lengthPerNgram + i]) {
				equal = 0;
				break;
			}
		}
		if (equal == 0 || !groupContext) {
			data[preNgramId * lengthPerNgram + n + 2] = ngramId + 1;
			preNgramId = ngramId + 1;
		}
	}
	if (equal == 1) {
		data[preNgramId * lengthPerNgram + n + 2] = ngramNumber;
	}
	data[ngramNumber * lengthPerNgram - 1] = ngramNumber;
	probTensor.resize(ngramNumber, 1);
	// for test
	//cout << "NgramNCEDataSet::createTensor dataTensor: " << endl;
	//dataTensor.write();
	//cout << "NgramNCEDataSet::createTensor probTensor: " << endl;
	//probTensor.info();
	return dataTensor;
}

int NgramNCEDataSet::readTextNgram(ioFile* iof) {
	string line;
	string invLine;
	int readLineNumber = 0;
	while (!iof->getEOF()) {
		if (iof->getLine(line)) {
			if (!checkBlankString(line)) {
				if (type == 0 || type == 2) {
					addLine(line);
				} else if (type == 1) {
					invLine = inverse(line);
					addLine(invLine);
				}
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

int NgramNCEDataSet::readTextNgramWithCoef(ioFile* iof) {
	string line;
	string invLine;
	int readLineNumber = 0;
	float currentCoef;
	while (!iof->getEOF()) {
		if (iof->getLine(line)) {
			if (!checkBlankString(line)) {
				currentCoef = getCoefFromString(line);
				if (!checkBlankString(line)) {
					if (type == 0 || type == 2) {
						addLineWithCoef(line, currentCoef);
					} else if (type == 1) {
						invLine = inverse(line);
						addLineWithCoef(invLine, currentCoef);
					}
				}
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

int NgramNCEDataSet::readCoBiNgram(ioFile* iof) {
	int readLineNumber = 0;
	int i;
	int N;
	int ngramNumberInFile;
	iof->readInt(ngramNumberInFile);
	iof->readInt(N);
	int readTextNgram[N];
	int offset = N - n;
	if (offset < 0) {
		cerr << "ERROR: order in id file is too small:" << N << " < " << n
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
		for (i = 0; i < n; i++) {
			data[ngramNumber * (n + 3) + i] = readTextNgram[offset + i];
		}
		data[ngramNumber * (n + 3) + n] = ID_END_NGRAM;
		data[ngramNumber * (n + 3) + n + 1] = ngramNumber;
		data[ngramNumber * (n + 3) + n + 2] = 0;
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

void NgramNCEDataSet::writeReBiNgram(ioFile* iof) {
	iof->writeInt(realNgramNumberAfterGrouping);
	iof->writeInt(n);
	int ngramId = 0;
	for (ngramId = 0; ngramId < ngramNumber; ngramId++) {
		// write only n-grams with non-zero coefficients
		if (coef[data[ngramId * lengthPerNgram + n + 1]] != 0) {
			iof->writeIntArray(data + ngramId * lengthPerNgram, n);
			iof->writeFloat(coef[data[ngramId * lengthPerNgram + n + 1]]);
		}
	}
}

string NgramNCEDataSet::inverse(string line) {
	istringstream streamLine(line);
	string word;
	string newLine = "";
	while (streamLine >> word) {
		newLine = word + " " + newLine;
	}
	return newLine;
}

int nce_compare(const void *ngram1, const void *ngram2) {

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

void NgramNCEDataSet::sortNgram() {
	// for test
	cout << "NgramNCEDataSet::sortNgram sort" << endl;
	if (sorted == 0) {
		qsort((void*) data, (size_t) ngramNumber,
				lengthPerNgram * sizeof(unsigned int), nce_compare);
	} else {
		cout << "NgramNCEDataSet::sortNgram data has been sorted" << endl;
	}
	sorted = 1;
}

void NgramNCEDataSet::shuffle(int times) {
	// for test
	cout << "NgramNCEDataSet::shuffle shuffle" << endl;
	// firstly, grouping n-grams
	this->realNgramNumberAfterGrouping = groupingNgram();
	int *tg = new int[lengthPerNgram * sizeof(int)];
	int i;
	int p1, p2;
	float tmp;
	for (i = 0; i < times * ngramNumber; i++) {
		p1 = (int) (ngramNumber * drand48());
		p2 = (int) (ngramNumber * drand48());
		memcpy(tg, data + p1 * lengthPerNgram, lengthPerNgram * sizeof(int));
		memcpy(data + p1 * lengthPerNgram, data + p2 * lengthPerNgram,
				lengthPerNgram * sizeof(int));
		memcpy(data + p2 * lengthPerNgram, tg, lengthPerNgram * sizeof(int));
	}
	sorted = 0;
}

int NgramNCEDataSet::writeReBiNgram() {
	int i;
	int ngramId = 0;
	string* inputWordsByIndex = new string[inputVoc->wordNumber];
	string* outputWordsByIndex = new string[outputVoc->wordNumber];
	int realNgramNumber = 0;
	for (ngramId = 0; ngramId < ngramNumber; ngramId++) {
		// write only n-grams with non-zero coefficients
		if (coef[data[ngramId * lengthPerNgram + n + 1]] != 0) {
			for (i = 0; i < n - 1; i++) {
				cout << inputWordsByIndex[data[ngramId * lengthPerNgram + i]] << " ";
			}
			cout << outputWordsByIndex[data[ngramId * lengthPerNgram + n - 1]] << " ";
			cout << data[ngramId * lengthPerNgram + n] << " ";
			cout << data[ngramId * lengthPerNgram + n + 1] << " ";
			cout << data[ngramId * lengthPerNgram + n + 2] << " ";
			cout << coef[data[ngramId * lengthPerNgram + n + 1]] << endl;
			realNgramNumber += 1;
		}
	}
	delete[] inputWordsByIndex;
	delete[] outputWordsByIndex;
	return realNgramNumber;
}

float NgramNCEDataSet::computePerplexity() {
	perplexity = 0;
	for (int i = 0; i < probTensor.length; i++) {
		// attention: coef and probTensor are in the same order (the order of n-gram before sortNgram() and shuffle())
		perplexity += -coef[i] * log(probTensor(i));
	}
	// for test
	cout << "NgramNCEDataSet::computePerplexity perplexity: " << perplexity
			<< endl;
	perplexity = perplexity / ngramNumber;
	return perplexity;
}

int NgramNCEDataSet::addLine(ioFile* iof) {
	int i = 0;
	string line;
	string headline;
	string invLine;
	string tailline;
	headline = "";
	tailline = "";
	float currentCoef = 0;
	// Normal
	if (type == 0) {
		for (i = 0; i < BOS; i++) {
			headline = headline + SS + " ";
		}
		tailline = tailline + " " + ES;
	}
	// Inverse
	else if (type == 1) {
		for (i = 0; i < BOS; i++) {
			tailline = tailline + " " + ES;
		}
		headline = headline + SS + " ";
	}
	// Center
	else if (type == 2) {
		for (i = 0; i < BOS / 2; i++) {
			tailline = tailline + " " + ES;
			headline = headline + SS + " ";
		}
	}
	if (iof->getLine(line)) {
		if (!checkBlankString(line)) {
			currentCoef = getCoefFromString(line);
			// line has been modified
			if (!checkBlankString(line)) {
				line = headline + line + tailline;
				if (type == 0 || type == 2) {
					addLineWithCoef(line, currentCoef);
				} else if (type == 1) {
					invLine = inverse(line);
					addLineWithCoef(invLine, currentCoef);
				}
			}
		}
	}
	return 1;
}

int NgramNCEDataSet::groupingNgram() {
	// rearrange the n-gram, sum coefficients over the same n-grams
	int realNgramNumber = 0;
	if (this->groupContext == 1) {
		// firstly, sort n-grams
		this->sortNgram();
		int ngramId = 0;
		int firstOccurOfNgram = ngramId;
		// sumOfCoef take the coef of firstOccurOfNgram
		float sumOfCoef = coef[data[firstOccurOfNgram * lengthPerNgram + n + 1]];
		int equal;
		int i;
		for (ngramId = 0; ngramId < ngramNumber - 1; ngramId++) {
			equal = 1;
			// compare n-grams, equal == 1 iff these are a same n-gram (context + word)
			for (i = 0; i < n; i++) {
				if (data[ngramId * lengthPerNgram + i]
						!= data[(ngramId + 1) * lengthPerNgram + i]) {
					equal = 0;
					break;
				}
			}
			if (equal == 0) {
				coef[data[firstOccurOfNgram * lengthPerNgram + n + 1]] =
						sumOfCoef;
				if (sumOfCoef != 0) {
					// a real n-gram detected
					// we eliminate all n-grams with coefficient 0, so realNgramNumber ++ only when sumOfCoef != 0
					realNgramNumber += 1;
				}
				// re-initialize sumOfCoef and firstOccurOfNgram will be the next n-gram
				firstOccurOfNgram = ngramId + 1;
				sumOfCoef =
						coef[data[firstOccurOfNgram * lengthPerNgram + n + 1]];
			} else {
				sumOfCoef += coef[data[(ngramId + 1) * lengthPerNgram + n + 1]];
				coef[data[(ngramId + 1) * lengthPerNgram + n + 1]] = 0;
			}
		}
		if (equal == 1) {
			coef[data[firstOccurOfNgram * lengthPerNgram + n + 1]] = sumOfCoef;
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
