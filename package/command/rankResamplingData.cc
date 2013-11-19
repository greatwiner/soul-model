#include "mainModel.H"
#include "time.h"
#include<algorithm>
#include <iterator>
int
main(int argc, char *argv[])
{
  if (argc != 11)
    {
      cout
          << "type dataDesFileName inputVocFileName outputVocFileName n mapIUnk mapOUnk prefixOutputFileName minEpochs maxEpochs"
          << endl;
      cout << "type = n(normal), i(inverse), m(middle)" << endl;
      return 0;
    }
  else
    {
      srand48(time(NULL));
      srand(time(NULL));
      char* typeStr = argv[1];
      int type;
      if (!strcmp(typeStr, "n"))
        {
          type = 0;
        }
      else if (!strcmp(typeStr, "i"))
        {
          type = 1;
        }
      else if (!strcmp(typeStr, "m"))
        {
          type = 2;
        }
      char* dataDesFileName = argv[2];
      char* inputVocFileName = argv[3];
      char* outputVocFileName = argv[4];
      int n = atoi(argv[5]);
      int mapIUnk = atoi(argv[6]);
      int mapOUnk = atoi(argv[7]);
      int BOS = n - 1;
      int times = 10;
      char* prefixOutputFileName = argv[8];
      int minEpochs = atoi(argv[9]);
      int maxEpochs = atoi(argv[10]);
      char outputFileName[260];
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
      NgramRankDataSet* dataSet;
      char * maxNgramNumberEnv;
      maxNgramNumberEnv = getenv("RESAMPLING_NGRAM_NUMBER");
      if (maxNgramNumberEnv != NULL)
        {
          dataSet = new NgramRankDataSet(type, n, BOS, inputVoc, outputVoc,
              mapIUnk, mapOUnk, atoi(maxNgramNumberEnv));
        }
      else
        {
          dataSet = new NgramRankDataSet(type, n, BOS, inputVoc, outputVoc,
              mapIUnk, mapOUnk, RESAMPLING_NGRAM_NUMBER);
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
              resampling = dataSet->resamplingDataDes(dataDesFileName, type);
            }
          cout << "shuffle epoch: " << iter << " with " << dataSet->ngramNumber
              << " ngrams" << endl;
          dataSet->shuffle(times);

          cout << "write to file : " << outputFileName << endl;
          iofO.takeWriteFile(outputFileName);
          dataSet->writeReBiNgram(&iofO);
        }
      delete dataSet;
      delete inputVoc;
      delete outputVoc;
      return 0;
    }
}
