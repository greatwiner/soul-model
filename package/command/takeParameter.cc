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
      iof->takeReadFile(inputFileName);
      model->baseNetwork->lkt->weight.read(iof);
      int i;
      cout << "Take base paras" << endl;
      if (model->recurrent == 0)
        {
          for (i = 0; i < model->baseNetwork->size; i += model->hiddenStep)
            {
              strcpy(inputFileName, prefixParas);
              strcat(inputFileName, "BaseWeight");
              sprintf(convert, "%d", i);
              strcat(inputFileName, convert);
              if (!iocf.check(inputFileName, 1))
                {
                  return 1;
                }
              iof->takeReadFile(inputFileName);
              model->baseNetwork->modules[i]->weight.read(iof);
              strcpy(inputFileName, prefixParas);
              strcat(inputFileName, "BaseBias");
              strcat(inputFileName, convert);
              if (!iocf.check(inputFileName, 1))
                {
                  return 1;
                }
              iof->takeReadFile(inputFileName);
              if (model->name == ROVN && i == 0)
                {
                  model->baseNetwork->modules[i]->vectorInput.read(iof);
                }
              else
                {
                  model->baseNetwork->modules[i]->bias.read(iof);
                }

            }
        }
      else
        {
          strcpy(inputFileName, prefixParas);
          strcat(inputFileName, "BaseWeight");
          sprintf(convert, "%d", 0);
          strcat(inputFileName, convert);
          if (!iocf.check(inputFileName, 1))
            {
              return 1;
            }
          iof->takeReadFile(inputFileName);
          model->baseNetwork->modules[0]->weight.read(iof);
          strcpy(inputFileName, prefixParas);
          strcat(inputFileName, "BaseBias");
          strcat(inputFileName, convert);
          if (!iocf.check(inputFileName, 1))
            {
              return 1;
            }
          iof->takeReadFile(inputFileName);
          model->baseNetwork->modules[0]->vectorInput.read(iof);

          for (i = 1; i < model->baseNetwork->size; i += model->hiddenStep)
            {
              strcpy(inputFileName, prefixParas);
              strcat(inputFileName, "BaseWeight");
              sprintf(convert, "%d", i);
              strcat(inputFileName, convert);
              if (!iocf.check(inputFileName, 1))
                {
                  return 1;
                }
              iof->takeReadFile(inputFileName);
              model->baseNetwork->modules[i]->weight.read(iof);
              strcpy(inputFileName, prefixParas);
              strcat(inputFileName, "BaseBias");
              strcat(inputFileName, convert);
              if (!iocf.check(inputFileName, 1))
                {
                  return 1;
                }
              iof->takeReadFile(inputFileName);
              model->baseNetwork->modules[i]->bias.read(iof);

            }
        }
      cout << "Take output paras" << endl;
      for (i = 0; i < model->outputNetworkNumber; i++)
        {
          strcpy(inputFileName, prefixParas);
          strcat(inputFileName, "OutputWeight");
          sprintf(convert, "%d", i);
          strcat(inputFileName, convert);
          if (!iocf.check(inputFileName, 1))
            {
              return 1;
            }
          iof->takeReadFile(inputFileName);
          model->outputNetwork[i]->weight.read(iof);
          strcpy(inputFileName, prefixParas);
          strcat(inputFileName, "OutputBias");
          strcat(inputFileName, convert);
          if (!iocf.check(inputFileName, 1))
            {
              return 1;
            }
          iof->takeReadFile(inputFileName);
          model->outputNetwork[i]->bias.read(iof);
        }

      ioFile oIof;
      oIof.takeWriteFile(outputModelFileName);
      model->write(&oIof, 1);

      delete iof;
      delete model;
    }
  return 0;
}

