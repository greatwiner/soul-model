#include "mainModel.H"
int
main(int argc, char *argv[])
{
  if (argc != 8)
    {
      cout
          << "inModelFileName newOutputVocFileName mapIUnk mapOUnk codeWordFileName outputNetworkSizeFileName outModelFileName"
          << endl;
      return 0;
    }
  else
    {
      srand48(time(NULL));
      srand(time(NULL));
      char* inModelFileName = argv[1];
      char* newOutputVocFileName = argv[2];
      int mapIUnk = atoi(argv[3]);
      int mapOUnk = atoi(argv[4]);
      char* codeWordFileName = argv[5];
      char* outputNetworkSizeFileName = argv[6];
      char* outModelFileName = argv[7];
      ioFile iofC;
      if (!iofC.check(inModelFileName, 1))
        {
          return 1;
        }
      if (!iofC.check(newOutputVocFileName, 1))
        {
          return 1;
        }

      if (!iofC.check(codeWordFileName, 0) && strcmp(codeWordFileName, "xxx"))
        {
          return 1;
        }
      if (!iofC.check(outputNetworkSizeFileName, 0) && strcmp(
          outputNetworkSizeFileName, "xxx"))
        {
          return 1;
        }
      if (iofC.check(outModelFileName, 0))
        {
          cerr << "Prototype exists" << endl;
          return 1;
        }
      ioFile readIof;
      readIof.format = BINARY;

      NeuralModel* inModel;
      READMODEL(inModel, 0, inModelFileName);

      inModel->mapIUnk = mapIUnk;
      inModel->mapOUnk = mapOUnk;
      //New outputVoc
      SoulVocab* oldOutputVoc = inModel->outputVoc;
      inModel->outputVoc = new SoulVocab(newOutputVocFileName);

      //New codeWord

      intTensor newCodeWord;

      if (!strcmp(codeWordFileName, "xxx"))
        {
          newCodeWord.resize(inModel->outputVoc->wordNumber, 2);
          newCodeWord = 0;
          for (int wordIndex = 0; wordIndex < inModel->outputVoc->wordNumber; wordIndex++)
            {
              newCodeWord(wordIndex, 1) = wordIndex;
            }
        }
      else
        {
          readIof.takeReadFile(codeWordFileName);
          newCodeWord.read(&readIof);
        }

      intTensor oldCodeWord;
      oldCodeWord.copy(inModel->codeWord);
      inModel->codeWord.resize(newCodeWord);
      inModel->codeWord.copy(newCodeWord);

      //New outputNetworkSize
      intTensor newOutputNetworkSize;

      if (!strcmp(outputNetworkSizeFileName, "xxx"))
        {
          newOutputNetworkSize.resize(1, 1);
          newOutputNetworkSize(0) = inModel->outputVoc->wordNumber;
        }
      else
        {
          readIof.takeReadFile(outputNetworkSizeFileName);
          newOutputNetworkSize.read(&readIof);
        }

      intTensor oldOutputNetworkSize;
      oldOutputNetworkSize.copy(inModel->outputNetworkSize);
      inModel->outputNetworkSize.resize(newOutputNetworkSize);
      inModel->outputNetworkSize.copy(newOutputNetworkSize);
      //New outputNetwork
      Module** oldOutputNetwork = inModel->outputNetwork;
      inModel->outputNetworkNumber = newOutputNetworkSize.size[0];

      inModel->outputNetwork = new Module*[inModel->outputNetworkNumber];
      LinearSoftmax* sl = new LinearSoftmax(inModel->hiddenLayerSize,
          inModel->outputNetworkSize(0), inModel->blockSize, inModel->otl);
      inModel->outputNetwork[0] = sl;
      int i;
      for (i = 1; i < inModel->outputNetworkNumber; i++)
        {
          sl = new LinearSoftmax(inModel->hiddenLayerSize,
              inModel->outputNetworkSize(i), 1, inModel->otl);
          inModel->outputNetwork[i] = sl;
        }
      inModel->doneForward.resize(inModel->outputNetworkNumber, 1);
      inModel->maxCodeWordLength = inModel->codeWord.size[1];
      inModel->localCodeWord.resize(inModel->blockSize,
          inModel->maxCodeWordLength);
      //Copy outputNetwork[0]
      floatTensor selectOutWeight;
      floatTensor selectInWeight;

      VocNode* run;
      int newIndex, oldIndex;
      for (i = 0; i < oldOutputVoc->tableSize; i++)
        {
          run = oldOutputVoc->table[i];
          while (run->next != NULL)
            {
              run = run->next;
              newIndex = inModel->codeWord(
                  inModel->outputVoc->index(run->word), 1);
              oldIndex = oldCodeWord(run->index, 1);
              if (newIndex != ID_UNK)
                {
                  selectInWeight.select(oldOutputNetwork[0]->weight, 1,
                      oldIndex);
                  selectOutWeight.select(inModel->outputNetwork[0]->weight, 1,
                      newIndex);
                  selectOutWeight.copy(selectInWeight);
                  inModel->outputNetwork[0]->bias(newIndex)
                      = oldOutputNetwork[0]->bias(oldIndex);
                }
            }
        }

      //New dataSet
      DataSet* oldDataSet = inModel->dataSet;
      inModel->dataSet = new NgramDataSet(inModel->ngramType, inModel->n,
          inModel->BOS, inModel->inputVoc, inModel->outputVoc,
          inModel->mapIUnk, inModel->mapOUnk, BLOCK_NGRAM_NUMBER);

      ioFile oIof;
      oIof.takeWriteFile(outModelFileName);
      inModel->write(&oIof, 1);

      delete oldOutputVoc;
      delete[] oldOutputNetwork;
      delete oldDataSet;
      delete inModel;
    }
  return 0;
}

