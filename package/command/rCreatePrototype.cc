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
  if (argc != 11)
    {
      cout
          << "type inputVocFileName outputVocFileName n dimensionSize nonLinearType"
          << " hiddenLayerSizeCode codeWordFileName outputNetworkSizeFileName outputModelFileName"
          << endl << "type = covr, ovr" << endl
          << "nonLinearType = t:Tanh, s:Sigmoid, l:linear" << endl;

      return 1;
    }
  else
    {
      srand48(time(NULL));
      srand(time(NULL));
      NeuralModel* modelPrototype;
      char* name = argv[1];
      char* inputVocFileName = argv[2];
      char* outputVocFileName = argv[3];
      int n = atoi(argv[4]);
      int dimensionSize = atoi(argv[5]);
      char* nonLinearType = argv[6];

      char* hiddenLayerSizeCode = argv[7];
      char* codeWordFileName = argv[8];
      char* outputNetworkSizeFileName = argv[9];
      char* outputModelFileName = argv[10];
      int blockSize = 1;
      ioFile iof;
      if (!iof.check(inputVocFileName, 1))
        {
          return 1;
        }
      if (!iof.check(outputVocFileName, 1))
        {
          return 1;
        }
      if (strcmp(codeWordFileName, "xxx"))
        {
          if (!iof.check(codeWordFileName, 1))
            {
              return 1;
            }
        }
      if (strcmp(outputNetworkSizeFileName, "xxx"))
        {
          if (!iof.check(outputNetworkSizeFileName, 1))
            {
              return 1;
            }
        }
      if (iof.check(outputModelFileName, 0))
        {
          cerr << "Prototype exists" << endl;
          return 1;
        }
      intTensor hiddenLayerSizeArray;
      getHiddenCode(hiddenLayerSizeCode, hiddenLayerSizeArray);
      modelPrototype = new RecurrentModel(name, inputVocFileName,
          outputVocFileName, blockSize, n, dimensionSize, nonLinearType,
          hiddenLayerSizeArray, codeWordFileName, outputNetworkSizeFileName);

      ioFile mIof;
      mIof.takeWriteFile(outputModelFileName);
      modelPrototype->write(&mIof, 1);

      delete modelPrototype;
      return 0;
    }
}
