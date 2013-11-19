#include "mainModel.H"
#include "time.h"
#include<algorithm>
#include <iterator>
int
main(int argc, char *argv[])
{
	// for test
	//cout << "rResamplingData::main here 5" << endl;
  if (argc != 10)
    {
      cout
          << "dataDesFileName inputVocFileName outputVocFileName n cont blockSize prefixOutputFileName minEpochs maxEpochs"
          << endl;
      return 0;
    }
  else
    {
      srand48(time(NULL));
      srand(time(NULL));
      char* dataDesFileName = argv[1];
      char* inputVocFileName = argv[2];
      char* outputVocFileName = argv[3];
      int n = atoi(argv[4]);
      int cont = atoi(argv[5]);
      int blockSize = atoi(argv[6]);
      char* prefixOutputFileName = argv[7];
      int minEpochs = atoi(argv[8]);
      int maxEpochs = atoi(argv[9]);
      char outputModelFileName[260];
      ioFile iofC;
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
      SoulVocab* inputVoc = new SoulVocab(inputVocFileName);
      SoulVocab* outputVoc = new SoulVocab(outputVocFileName);
      DataSet* dataSet;
      char * maxNgramNumberEnv;
      maxNgramNumberEnv = getenv("RESAMPLING_NGRAM_NUMBER");
      if (maxNgramNumberEnv != NULL)
        {
          dataSet = new RecurrentDataSet(n, inputVoc, outputVoc, cont,
              blockSize, atoi(maxNgramNumberEnv));
        }
      else
        {

          dataSet = new RecurrentDataSet(n, inputVoc, outputVoc, cont,
              blockSize, RESAMPLING_NGRAM_NUMBER);
        }
      if (dataSet->data == NULL)
        {
          return 1;
        }
      int resampling = 1;
      ioFile iofO;
      // for test
      //cout << "rResamplingData::main here 4" << endl;
      for (int iter = minEpochs; iter <= maxEpochs; iter++)
        {
          strcpy(outputModelFileName, prefixOutputFileName);
          ostringstream iConvert;
          iConvert << iter;
          strcat(outputModelFileName, iConvert.str().c_str());
          //for test
          //cout << "rResamplingData::main here" << endl;
          if (iofC.check(outputModelFileName, 0))
            {
              cout << "file: " << outputModelFileName << " exists" << endl;
              continue;
            }
          if (resampling)
            {
        	  // for test
        	  //cout << "rResamplingData::main here 1" << endl;
              resampling = dataSet->resamplingDataDes(dataDesFileName, 0);
            }
          // for test
          //cout << "rResamplingData::main here 2" << endl;
          dataSet->createTensor();
          // for test
          //cout << "rResamplingData::main here 3" << endl;
          cout << "create data with (not exact) " << dataSet->ngramNumber
              << " ngrams" << endl;

          cout << "write to file : " << outputModelFileName << endl;
          iofO.takeWriteFile(outputModelFileName);
          dataSet->writeReBiNgram(&iofO);
        }
      delete dataSet;
      delete inputVoc;
      delete outputVoc;
      return 0;
    }
}
