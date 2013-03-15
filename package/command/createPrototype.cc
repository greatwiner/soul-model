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
  if (argc != 13)
    {
      cout << "type inputVocFileName outputVocFileName mapIUnk mapOUnk"
          << " n dimensionSize nonLinearType"
          << " hiddenLayerSizeCode codeWordFileName outputNetworkSizeFileName outputModelFileName"
          << endl;
      cout << "type = cn, ovn, ovnb, rovn, maxovn, lbl" << endl;
      cout << "nonLinearType = l (linear), s (sigmoid), t (tangent hyperbolic)"
          << endl;
      cout
          << "hiddenLayerSizeCode, e.g., 1000_500: 2 hidden layers of 1000, 500 nodes"
          << endl;
      return 1;
    }
  else
    {
      srand48(time(NULL));
      srand(time(NULL));
      NeuralModel* modelPrototype;
      string name = argv[1];
      if (name != CN && name != OVN && name != OVNB && name != MAXOVN && name != ROVN && name
          != LBL)
        {
          cerr << "Which model do you want?" << endl;
          return 1;
        }

      char* contextVocFileName = argv[2];
      char* predictVocFileName = argv[3];
      int mapIUnk = atoi(argv[4]);
      int mapOUnk = atoi(argv[5]);
      int n = atoi(argv[6]);
      int BOS = n - 1;
      int dimensionSize = atoi(argv[7]);

      string nonLinearType = argv[8];
      if (nonLinearType != TANH && nonLinearType != SIGM && nonLinearType
          != LINEAR)
        {
          cerr << "Which activation do you want?" << endl;
          return 1;
        }

      char* hiddenLayerSizeCode = argv[9];
      char* codeWordFileName = argv[10];
      char* outputNetworkSizeFileName = argv[11];
      char* outputModelFileName = argv[12];
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
      ioFile mIof;
      // Bayesian model
      if (name == OVNB) {
		  NgramModel_Bayes* modelPrototypeBayes = new NgramModel_Bayes(name, contextVocFileName,
				  predictVocFileName, mapIUnk, mapOUnk, BOS, blockSize, n,
				  dimensionSize, nonLinearType, hiddenLayerSizeArray, codeWordFileName,
				  outputNetworkSizeFileName);
		  mIof.takeWriteFile(outputModelFileName);
		  modelPrototypeBayes->write(&mIof);
		  delete modelPrototypeBayes;
      }
      else {
    	  modelPrototype = new NgramModel(name, contextVocFileName,
    	      	            predictVocFileName, mapIUnk, mapOUnk, BOS, blockSize, n,
    	      	            dimensionSize, nonLinearType, hiddenLayerSizeArray, codeWordFileName,
    	      	            outputNetworkSizeFileName);
		  mIof.takeWriteFile(outputModelFileName);
		  modelPrototype->write(&mIof);
		  delete modelPrototype;
      }
      return 0;
    }

}

