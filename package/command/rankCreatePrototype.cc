#include "mainModel.H"
void
getHiddenCode(char* input, intTensor& outputTensor)
{
  char hidden[260];
  strcpy(hidden, input);
  int hiddenNumber = 1;
  for (int i = 0; i < strlen(input); i++)
    {
      if (hidden[i] == '_')
        {
          hidden[i] = ' ';
          hiddenNumber++;
        }
    }
  string strHidden = hidden;
  istringstream streamHidden(strHidden);
  string word;
  outputTensor.resize(hiddenNumber, 1);
  hiddenNumber = 0;
  while (streamHidden >> word)
    {
      outputTensor(hiddenNumber) = atoi(word.c_str());
      hiddenNumber++;
    }
}
int
main(int argc, char *argv[])
{
  //Create model
  if (argc != 12)
    {
      cout
          << "type ngramType inputVocFileName outputVocFileName mapIUnk mapOUnk "
          << " n dimensionSize nonLinearType"
          << " hiddenLayerSizeCode outputModelFileName" << endl;
      cout << "type = rankovn, rankcn" << endl;
      cout << "ngramType = n(normal), i(inverse), m(middle)" << endl;
      return 1;
    }
  else
    {
      srand48(time(NULL));
      srand(time(NULL));
      NeuralModel* modelPrototype;

      string name = argv[1];
      if (name != RANKOVN && name != RANKCN)
        {
          cerr << "Which model do you want?" << endl;
          return 1;
        }

      char* ngramTypeStr = argv[2];
      int ngramType;
      if (!strcmp(ngramTypeStr, "n"))
        {
          ngramType = 0;
        }
      else if (!strcmp(ngramTypeStr, "i"))
        {
          ngramType = 1;
        }
      else if (!strcmp(ngramTypeStr, "m"))
        {
          ngramType = 2;
        }

      char* contextVocFileName = argv[3];
      char* predictVocFileName = argv[4];
      int mapIUnk = atoi(argv[5]);
      int mapOUnk = atoi(argv[6]);
      int n = atoi(argv[7]);
      int BOS = n - 1;
      int dimensionSize = atoi(argv[8]);
      string nonLinearType = argv[9];
      if (nonLinearType != TANH && nonLinearType != SIGM && nonLinearType
          != LINEAR)
        {
          cerr << "Which activation do you want?" << endl;
          return 1;
        }

      char* hiddenLayerSizeCode = argv[10];
      char* outputModelFileName = argv[11];
      int blockSize = 1;
      ioFile iof;
      if (!iof.check(contextVocFileName, 1))
        {
          return 1;
        }
      if (!iof.check(predictVocFileName, 1))
        {
          return 1;
        }
      if (iof.check(outputModelFileName, 0))
        {
          cerr << "Prototype exists" << endl;
          return 1;
        }
      intTensor hiddenLayerSizeArray;
      getHiddenCode(hiddenLayerSizeCode, hiddenLayerSizeArray);
      modelPrototype = new NgramRankModel(name, ngramType, contextVocFileName,
          predictVocFileName, mapIUnk, mapOUnk, BOS, blockSize, n,
          dimensionSize, nonLinearType, hiddenLayerSizeArray);
      ioFile mIof;
      mIof.takeWriteFile(outputModelFileName);
      modelPrototype->write(&mIof, 1);

      delete modelPrototype;
      return 0;
    }
}
