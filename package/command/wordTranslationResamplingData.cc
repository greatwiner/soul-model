#include "mainModel.H"
#include "time.h"
#include<algorithm>
#include <iterator>
int main(int argc, char *argv[]) {
	if (argc != 12) {
		cout
				<< "ngramType dataDesFileName inputVocFileName outputVocFileName n mapIUnk mapOUnk prefixOutputFileName minEpoch maxEpoch modelType"
				<< endl;
		cout << "ngramType 0: trgSrc, 1: trg, 2: srcTrg, 3: src" << endl;
		cout << "modelType: dwtovn , dwtovn_nce" << endl;
		return 0;
	} else {
		srand48 (time(NULL));srand(time(NULL));
		int ngramType = atoi(argv[1]);
		char* dataDesFileName = argv[2];
		char* inputVocFileName = argv[3];
		char* outputVocFileName = argv[4];
		int n = atoi(argv[5]);
		int mapIUnk = atoi(argv[6]);
		int mapOUnk = atoi(argv[7]);
		int BOS = n;
		int times = 10;
		char* prefixOutputFileName = argv[8];
		int minEpochs = atoi(argv[9]);
		int maxEpochs = atoi(argv[10]);
		string modelType = argv[11];
		char outputModelFileName[260];
		ioFile iofC;
		int realNgramNumber = 0;
		// for test
		//cout << "wordTranslationResamplingData::main here 6" << endl;
		if (!iofC.check(dataDesFileName, 1))
		{
			return 1;
		}
		if (!iofC.check(inputVocFileName, 1))
		{
			return 1;
		}
		if (!iofC.check(outputVocFileName, 1))
		{
			return 1;
		}
		// for test
		//cout << "wordTranslationResamplingData::main here 7" << endl;
		SoulVocab* inputVoc = new SoulVocab(inputVocFileName);
		// for test
		//cout << "wordTranslationResamplingData::main here 8" << endl;
		SoulVocab* outputVoc = new SoulVocab(outputVocFileName);
		// for test
		//cout << "wordTranslationResamplingData::main here 9" << endl;
		DataSet* dataSet;
		char * maxNgramNumberEnv;
		maxNgramNumberEnv = getenv("RESAMPLING_NGRAM_NUMBER");
		if (maxNgramNumberEnv != NULL)
		{
			// for test
			//cout << "wordTranslationResamplingData::main here 10" << endl;
			if (modelType == WTOVN) {
				dataSet = new NgramWordTranslationDataSet(ngramType, n, BOS,
						inputVoc, outputVoc, mapIUnk, mapOUnk, atoi(maxNgramNumberEnv));
			} else if (modelType == WTOVN_NCE) {
				dataSet = new NgramNCEWordTranslationDataSet(ngramType, n, BOS,
						inputVoc, outputVoc, mapIUnk, mapOUnk, atoi(maxNgramNumberEnv));
			}
			else {
				cout << "What type of model are you training?" << endl;
				exit(1);
			}
		}
		else
		{
			// for test
			//cout << "wordTranslationResamplingData::main here 11" << endl;
			if (modelType == WTOVN) {
				dataSet = new NgramWordTranslationDataSet(ngramType, n, BOS,
						inputVoc, outputVoc, mapIUnk, mapOUnk, RESAMPLING_NGRAM_NUMBER);
			} else if (modelType == WTOVN_NCE) {
				dataSet = new NgramNCEWordTranslationDataSet(ngramType, n, BOS,
						inputVoc, outputVoc, mapIUnk, mapOUnk, RESAMPLING_NGRAM_NUMBER);
			} else {
				cout << "What type of model are you training?" << endl;
				exit(1);
			}
			// for test
			//cout << "wordTranslationResamplingData::main here 13" << endl;
		}
		if (dataSet->data == NULL)
		{
			return 1;
		}
		int resampling = 1;
		ioFile iofO;
		// for test
		//cout << "wordTranslationResamplingData::main here 12" << endl;
		for (int iter = minEpochs; iter <= maxEpochs; iter++)
		{
			strcpy(outputModelFileName, prefixOutputFileName);
			ostringstream iConvert;
			iConvert << iter;
			strcat(outputModelFileName, iConvert.str().c_str());
			if (iofC.check(outputModelFileName, 0))
			{
				cout << "file: " << outputModelFileName << " exists" << endl;
				continue;
			}
			if (resampling)
			{
				// for test
				//cout << "wordTranslationResamplingData::main here" << endl;
				resampling = dataSet->resamplingDataDes(dataDesFileName,
						ngramType);
				// for test
				//cout << "wordTranslationResamplingData::main dataSet after resampling: " << endl;
				//realNgramNumber = dataSet->writeReBiNgram();
				//cout << "wordTranslationResamplingData::main realNgramNumber: " << realNgramNumber << endl;
			}
			cout << "shuffle epoch: " << iter << " with " << dataSet->ngramNumber
			<< " ngrams" << endl;
			// for test
			//cout << "wordTranslationResamplingData::main here2" << endl;
			dataSet->shuffle(times);
			// for test
			//cout << "wordTranslationResamplingData::main dataSet after shuffle: " << endl;
			//realNgramNumber = dataSet->writeReBiNgram();
			//cout << "wordTranslationResamplingData::main realNgramNumber: " << realNgramNumber << endl;

			cout << "write to file : " << outputModelFileName << endl;
			iofO.takeWriteFile(outputModelFileName);
			// for test
			//cout << "wordTranslationResamplingData::main here4" << endl;
			dataSet->writeReBiNgram(&iofO);
			// for test
			//cout << "wordTranslationResamplingData::main here5" << endl;
		}
		delete dataSet;
		delete inputVoc;
		delete outputVoc;
		return 0;
	}
}
