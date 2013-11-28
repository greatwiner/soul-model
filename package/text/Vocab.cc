#include "text.H"
//Function of VocNode
VocNode::VocNode() {
	index = ID_INIT;
	next = NULL;
}
VocNode::VocNode(string word, int index) {
	this->word = word;
	this->index = index;
	next = NULL;
}
VocNode::~VocNode() {
	if (next != NULL) {
		delete next;
		next = NULL;
	}
}
//Function of SoulVocab
SoulVocab::SoulVocab() {
	tableSize = VOCAB_TABLE_SIZE;
	wordNumber = 0;
	int i;
	table = new VocNode*[tableSize];
	runTable = new VocNode*[tableSize];
	for (i = 0; i < tableSize; i++) {
		table[i] = new VocNode("VOC_ROOT", ID_ROOT);
		table[i]->next = NULL;
		runTable[i] = table[i];
	}
}

SoulVocab::SoulVocab(char* dataFileName, char* indexFileName) {
	tableSize = VOCAB_TABLE_SIZE;
	wordNumber = 0;
	int i;
	table = new VocNode*[tableSize];
	runTable = new VocNode*[tableSize];
	for (i = 0; i < tableSize; i++) {
		table[i] = new VocNode("VOC_ROOT", ID_ROOT);
		table[i]->next = NULL;
		runTable[i] = table[i];
	}

	string line;
	int readLineNumber = 0;
	int lindex;
	ioFile iof;
	iof.format = TEXT;
	iof.takeReadFile(dataFileName);
	ioFile iofI;
	iofI.format = TEXT;
	iofI.takeReadFile(indexFileName);

	while (!iof.getEOF()) {
		if (iof.getLine(line)) {
			iofI.readInt(lindex);
			add(line, lindex);
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
	ss = index(SS);
	es = index(ES);
	unk = index(UNK);
}
SoulVocab::SoulVocab(char* dataFileName) {
	tableSize = VOCAB_TABLE_SIZE;
	wordNumber = 0;
	int i;
	table = new VocNode*[tableSize];
	runTable = new VocNode*[tableSize];
	for (i = 0; i < tableSize; i++) {
		table[i] = new VocNode("VOC_ROOT", ID_ROOT);
		table[i]->next = NULL;
		runTable[i] = table[i];
	}
	string line;
	int readLineNumber = 0;
	ioFile iof;
	iof.format = TEXT;
	iof.takeReadFile(dataFileName);
	while (!iof.getEOF()) {
		if (iof.getLine(line)) {
			add(line, readLineNumber);
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
	ss = index(SS);
	es = index(ES);
	unk = index(UNK);
}

SoulVocab::SoulVocab(SoulVocab* inSoulVocab) {
	tableSize = VOCAB_TABLE_SIZE;
	wordNumber = 0;
	int i;
	table = new VocNode*[tableSize];
	runTable = new VocNode*[tableSize];
	for (i = 0; i < tableSize; i++) {
		table[i] = new VocNode("VOC_ROOT", ID_ROOT);
		table[i]->next = NULL;
		runTable[i] = table[i];
	}
	VocNode* run;
	string word;
	for (i = 0; i < inSoulVocab->tableSize; i++) {
		run = inSoulVocab->table[i];
		while (run->next != NULL) {
			run = run->next;
			add(run->word, run->index, i);
		}
	}
	ss = index(SS);
	es = index(ES);
	unk = index(UNK);
}

void SoulVocab::read(ioFile* iof) {
	// for test
	//cout << "SoulVocab::read here" << endl;
	string localStr;
	iof->readString(localStr);
	// for test
	//cout << "SoulVocab::read localStr: " << localStr << endl;
	if (localStr != "voc") {
		cerr << "ERROR:voc header is not found" << endl;
		exit(1);
	}
	iof->readInt(tableSize);
	wordNumber = 0;
	int i;
	if (VOCAB_TABLE_SIZE < tableSize) {
		cerr << "ERROR: tableSize > VOC_TABLE_SIZE" << endl;
		exit(1);
	}

	intTensor info;
	info.read(iof);
	// for test
	//cout << "SoulVocab::read info: " << endl;
	//info.write();
	int iId = 0;
	int wordNo;
	VocNode* add;
	int j;
	// for test
	//cout << "SoulVocab::read runTable: " << endl;
	while (iId < info.size[0]) {
		i = info(iId);
		iId++;
		wordNo = info(iId);
		iId++;
		// for test
		//cout << "SoulVocab::read i: " << i << " wordNo: " << wordNo << " ";
		for (j = iId; j < iId + wordNo; j++) {
			wordNumber++;
			add = new VocNode();
			runTable[i]->next = add;
			add->index = info(j);
			// for test
			//cout << add->index << " : ";
			iof->readString(add->word);
			// for test
			//cout << add->word << " -> ";
			runTable[i] = add;
		}
		//cout << endl;
		iId = j;
	}
	// Read ~~~, for mmap
	iof->readString(localStr);
	ss = index(SS);
	es = index(ES);
	unk = index(UNK);
}

SoulVocab::~SoulVocab() {
	if (table != NULL) {
		int i;
		for (i = 0; i < tableSize; i++) {
			delete table[i];
		}
		delete[] table;
	}
	if (runTable != NULL) {
		delete[] runTable;
	}
}
int SoulVocab::getHashValue(string word) {
	int hashValue = 0;
	int idWord;
	for (idWord = 0; idWord < word.size(); idWord++) {
		hashValue = (256 * hashValue + (unsigned short) word[idWord])
				% tableSize;
	}
	return hashValue;
}

int SoulVocab::add(string word, int idWord) {
	int i;
	if (index(word) != ID_UNK) {
		return 0;
	}
	for (i = 0; i < word.length(); i++) {
		if (word[i] == ' ') {
			word[i] = '#';
		}
	}
	int hashValue = getHashValue(word);
	VocNode* add = new VocNode(word, idWord);
	runTable[hashValue]->next = add;
	runTable[hashValue] = add;
	wordNumber++;
	return 1;
}

int SoulVocab::add(string word, int idWord, int hashValue) {
	int i;
	if (index(word) != ID_UNK) {
		return 0;
	}
	for (i = 0; i < word.length(); i++) {
		if (word[i] == ' ') {
			word[i] = '#';
		}
	}
	VocNode* add = new VocNode(word, idWord);
	runTable[hashValue]->next = add;
	runTable[hashValue] = add;
	wordNumber++;
	return 1;
}

int SoulVocab::index(string inWord)          //Get index
		{
	string word;
	word = inWord;
	int i;
	for (i = 0; i < word.length(); i++) {
		if (word[i] == ' ') {
			word[i] = '#';
		}
	}
	int hashValue = getHashValue(word);

	VocNode* run;
	run = table[hashValue];
	while (run->word != word && run->next != NULL) {
		run = run->next;
	}
	if (run->word == word) {
		return run->index;
	} else {
		return ID_UNK;
	}
}

void SoulVocab::write(ioFile* iof) {
	// for test
	//cout << "SoulVocab::write here" << endl;
	iof->writeString("voc");
	// for test
	//cout << "SoulVocab::write here 1" << endl;
	iof->writeInt(tableSize);
	// for test
	//cout << "SoulVocab::write tableSize: " << tableSize << endl;
	VocNode* run;
	//Format info: [id_in_table number_of_words word1 word2]+, number_of_words <> 0
	intTensor info;
	info.resize(wordNumber + tableSize * 2, 1);
	int i;
	int iId = 0;
	int preId = 1;
	int realSize = wordNumber;
	int wordNo;
	for (i = 0; i < tableSize; i++) {
		// for test
		//cout << "SoulVocab::write i: " << i << endl;
		run = table[i];
		if (run->next != NULL) {
			realSize += 2;
			info(iId) = i;
			// for test
			//cout << "SoulVocab::write info(" << iId << ") = " << i << endl;
			iId++;
			preId = iId;
			iId++;
			wordNo = 0;
			// for test
			//cout << "SoulVocab::write here 1.1" << endl;
			while (run->next != NULL) {
				// for test
				//cout << "SoulVocab::write run->next: " << run->next << endl;
				run = run->next;
				info(iId) = run->index;
				// for test
				//cout << "SoulVocab::write info(" << iId << ") = " << run->index << endl;
				iId++;
				wordNo++;
			}
			info(preId) = wordNo;
			// for test
			//cout << "SoulVocab::write info(" << preId << ") = " << wordNo << endl;
		}
	}
	// for test
	//cout << "SoulVocab::write here 2" << endl;
	info.size[0] = realSize;
	info.length = realSize;
	info.write(iof);
	// for test
	//cout << "SoulVocab::write here 3" << endl;
	int addByte = wordNumber % sizeof(int);
	for (i = 0; i < tableSize; i++) {
		// for test
		//cout << "SoulVocab::write i: " << i << endl;
		run = table[i];
		while (run->next != NULL) {
			run = run->next;
			iof->writeString(run->word);
			addByte += run->word.size();
		}
	}
	addByte = 2 * sizeof(int) - addByte % sizeof(int) - 1;
	// for test
	//cout << "SoulVocab::write here 4" << endl;
	string localStr = "";
	for (i = 0; i < addByte; i++) {
		localStr = localStr + "~";
	}
	// for test
	//cout << "SoulVocab::write here 5" << endl;
	iof->writeString(localStr);
	// for test
	//cout << "SoulVocab::write here 6" << endl;
}

void SoulVocab::getWordByIndex(string* arrayOfString) {
	if (arrayOfString == NULL) {
		cout << "SoulVocab::getWordByIndex memory has not been allocated"
				<< endl;
		exit(1);
	}
	VocNode* run;
	int i = 0;
	for (i = 0; i < tableSize; i++) {
		run = table[i];
		while (run->next != NULL) {
			run = run->next;
			arrayOfString[run->index] = run->word;
		}
	}
}
