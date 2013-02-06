#include "mainModel.H"
int
main(int argc, char *argv[])
{
  if (argc != 4)
    {
      cout << "modelFileName prefixParas outputModelFileName" << endl;
      return 0;
    }
  if (argc == 4)
    {
      char* modelFileName = argv[1];
      char* prefixParas = argv[2];
      char* outputModelFileName = argv[3];

      ioFile iocf;
      if (!iocf.check(modelFileName, 1))
        {
          return 1;
        }
      if (iocf.check(outputModelFileName, 0))
        {
          cerr << "model file exists" << endl;
          return 1;
        }

      char convert[10];

      NeuralModel* model;
      READMODEL(model, 0, modelFileName);
      ioFile* iof = new ioFile();
      char inputFileName[260];
      strcpy(inputFileName, prefixParas);
      strcat(inputFileName, "LookupTable");
      if (!iocf.check(inputFileName, 1))
        {
          cout << "Can not file LookupTable" << endl;
          return 1;
        }
      iof->takeReadFile(inputFileName);
      model->baseNetwork->lkt->weight.read(iof);
      int i;
      for (i = 0; i < model->baseNetwork->size; i += model->hiddenStep)
        {
          strcpy(inputFileName, prefixParas);
          strcat(inputFileName, "BaseWeight");
          sprintf(convert, "%d", i);
          strcat(inputFileName, convert);
          if (!iocf.check(inputFileName, 1))
            {
              cout << "Can not file BaseWeight " << i << endl;
              break;
            }
          iof->takeReadFile(inputFileName);
          model->baseNetwork->modules[i]->weight.read(iof);
          strcpy(inputFileName, prefixParas);
          strcat(inputFileName, "BaseBias");
          strcat(inputFileName, convert);
          if (!iocf.check(inputFileName, 1))
            {
              cout << "Can not file BaseBias " << i << endl;
              break;
            }
          iof->takeReadFile(inputFileName);
          model->baseNetwork->modules[i]->bias.read(iof);
        }
      ioFile oIof;
      oIof.takeWriteFile(outputModelFileName);
      model->write(&oIof);

      delete iof;
      delete model;
    }
  return 0;
}

