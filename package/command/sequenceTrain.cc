#include "mainModel.H"

int
sequenceTrain(char* prefixModel, char* prefixData, int maxExampleNumber, char* trainingFileName,
    char* validationFileName, string validType, string learningRateType,
    int minIteration, int maxIteration)
{
  outils otl;
  char inputModelFileName[260];
  char convertStr[260];
  int iteration;
  int gz = 0;
  for (iteration = maxIteration; iteration >= minIteration - 2; iteration--)
    {
      sprintf(convertStr, "%ld", iteration);
      strcpy(inputModelFileName, prefixModel);
      strcat(inputModelFileName, convertStr);
      ioFile iof;
      if (!iof.check(inputModelFileName, 0))
        {
          strcat(inputModelFileName, ".gz");
          if (iof.check(inputModelFileName, 0))
            {
              gz = 1;
              break;
            }
        }
      else
        {
          gz = 0;
          break;
        }
    }
  if (iteration == minIteration - 3)
    {
      cerr << "Can not find training model " << minIteration - 1 << endl;
      return 1;
    }
  else if (iteration == maxIteration)
    {
      cerr << "All is done" << endl;
      return 1;
    }

  sprintf(convertStr, "%ld", iteration);
  strcpy(inputModelFileName, prefixModel);
  strcat(inputModelFileName, convertStr);
  if (gz)
    {
      strcat(inputModelFileName, ".gz");
    }
  NeuralModel* model;
  READMODEL(model, 0, inputModelFileName);

  model->sequenceTrain(prefixModel, gz, prefixData, maxExampleNumber, trainingFileName,
      validationFileName, validType, learningRateType, iteration + 1,
      maxIteration);
  delete model;
  return 0;
}

int
main(int argc, char *argv[])
{
  if (argc != 10)
    {
      cout
          << "prefixModel prefixData maxExampleNumber trainingFileName, validationFileName validType learningRateType minIteration maxIteration"
          << endl;
      cout << "validType: n(normal-text), l(ngram list), id (binary id ngram)"
          << endl;
      return 0;
    }
  char* prefixModel = argv[1];
  char* prefixData = argv[2];
  int maxExampleNumber = atoi(argv[3]);
  char* trainingFileName = argv[4];
  char* validationFileName = argv[5];
  string validType = argv[6];
  if (validType != "n" && validType != "l" && validType != "id")

    {
      cerr << "Which validType do you want?" << endl;
      return 1;
    }

  string learningRateType = argv[7];
  if (learningRateType != LEARNINGRATE_NORMAL && learningRateType
      != LEARNINGRATE_DOWN)
    {
      cerr << "Which learningRateType do you want?" << endl;
      return 1;
    }
  int minIteration = atoi(argv[8]);
  int maxIteration = atoi(argv[9]);
  srand(time(NULL));
  sequenceTrain(prefixModel, prefixData, maxExampleNumber, trainingFileName, validationFileName,
      validType, learningRateType, minIteration, maxIteration);
  return 0;
}

