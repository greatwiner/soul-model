#include "mainModel.H"
#include "time.h"
#include<algorithm>
#include <iterator>
int
main(int argc, char *argv[])
{
  if (argc != 6)
    {
      cout << "dataDesFileName n prefixOutputFileName minEpoch maxEpoch"
          << endl;
      return 0;
    }
  else
    {
      srand48(time(NULL));
      srand(time(NULL));
      char* dataDesFileName = argv[1];
      int n = atoi(argv[2]);
      int times = 10;
      char* prefixOutputFileName = argv[3];
      int minEpochs = atoi(argv[4]);
      int maxEpochs = atoi(argv[5]);
      char outputFileName[260];
      ioFile iofC;
      if (!iofC.check(dataDesFileName, 1))
        {
          return 1;
        }
      NgramDataSet* dataSet;
      char * maxNgramNumberEnv;
      maxNgramNumberEnv = getenv("RESAMPLING_NGRAM_NUMBER");
      if (maxNgramNumberEnv != NULL)
        {
          dataSet = new NgramDataSet(n, atoi(maxNgramNumberEnv));
        }
      else
        {
          dataSet = new NgramDataSet(n, RESAMPLING_NGRAM_NUMBER);
        }
      if (dataSet->data == NULL)
        {
          return 1;
        }
      int resampling = 1;
      ioFile iofO;
      for (int iter = minEpochs; iter <= maxEpochs; iter++)
        {
          strcpy(outputFileName, prefixOutputFileName);
          ostringstream iConvert;
          iConvert << iter;
          strcat(outputFileName, iConvert.str().c_str());
          if (iofC.check(outputFileName, 0))
            {
              cout << "file: " << outputFileName << " exists" << endl;
              continue;
            }
          if (resampling)
            {
              resampling = dataSet->resamplingIdDataDes(dataDesFileName);
            }
          cout << "shuffle epoch: " << iter << " with " << dataSet->ngramNumber
              << " ngrams" << endl;
          dataSet->shuffle(times);

          cout << "write to file : " << outputFileName << endl;
          iofO.takeWriteFile(outputFileName);
          dataSet->writeReBiNgram(&iofO);
        }
      delete dataSet;
      return 0;
    }
}
