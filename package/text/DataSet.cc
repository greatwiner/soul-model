#include "text.H"

DataSet::~DataSet() {
	if (data != NULL) {
		delete[] data;
	}
}

DataSet::DataSet() {
	groupContext = 1;
	data = NULL;
	ngramNumber = 0;
	maxNgramNumber = 0;
	this->realNgramNumberAfterGrouping = 0;
}

void DataSet::reset() {
	ngramNumber = 0;
}

int DataSet::checkBlankString(string line) {
	for (int i = 0; i < line.length(); i++) {
		if (line[i] != ' ') {
			return 0;
		}
	}
	return 1;
}

float DataSet::getCoefFromString(string& line) {
	string newLine = "";
	string word;
	string preWord;
	float currentCoef;
	istringstream streamLine(line);
	streamLine >> preWord;
	if (streamLine >> word) {
		do {
			if (atof(word.c_str())) {
				currentCoef = (float) atof(word.c_str());
			}
			newLine = newLine + " " + preWord;
			preWord = word;
		} while (streamLine >> word);
	} else {
		if (atof(preWord.c_str())) {
			currentCoef = (float) atof(preWord.c_str());
		} else {
			cout
					<< "DataSet::getCoefFromString line is not correctly formated with coef"
					<< endl;
			exit(0);
		}
	}
	line = newLine;
	return currentCoef;
}

int DataSet::resamplingDataDes(char* dataDesFileName, int type) {
	this->type = type;
	reset();
	ioFile iofRead;
	iofRead.format = TEXT;
	int resampling = 0;
	int allLineNumber = 0;
	string line;
	int totalLineNumber = 0;
	float percent;
	char dataFileName[260];
	int resamplingLineNumber;

	//Now read
	ioFile iof;
	iof.format = TEXT;
	iofRead.takeReadFile(dataDesFileName);
	while (!iofRead.getEOF()) {
		if (iofRead.getLine(line) && !checkBlankString(line)) {
			istringstream ostr(line);
			ostr >> dataFileName >> totalLineNumber >> percent;
			//cout << "DataSet::resamplingDataDes line: " << line << endl;
			if (percent < 1) {
				resampling = 1;
			}

			resamplingLineNumber = (int) (totalLineNumber * percent);
			if (!iof.check(dataFileName, 1)) {
				return 1;
			}
			iof.takeReadFile(dataFileName);
			cout << "read file: " << dataFileName << endl;
			resamplingText(&iof, totalLineNumber, resamplingLineNumber);
		}
	}
	return resampling;
}

int DataSet::eosLine(string line) {
	// for test
	//cout << "DataSet::eosLine line: " << endl;
	istringstream streamLine(line);
	string word;
	if (streamLine >> word) {
		// for test
		//cout << "DataSet::eosLine word: " << word << endl;
		return word == "EOS";
	} else {
		return 0;
	}
}
