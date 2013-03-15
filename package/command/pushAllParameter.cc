#include "mainModel.H"
int
main(int argc, char *argv[])
{
  if (argc != 4)
    {
      cout << "modelFileName prefixParas what" << endl;
      cout << "what: l, b, o, c, v, a" << endl;
      cout << "l: look-uptable" << endl;
      cout << "b: baseNetwork" << endl;
      cout << "o: outputNetwork" << endl;
      cout
          << "c: code (codeWord and outputNetworkSize) encoding a tree structure"
          << endl;
      cout << "v: vocabulary" << endl;
      cout << "a: all" << endl;
      return 0;
    }
  else
    {
      char* modelFileName = argv[1];
      char* prefixParas = argv[2];
      char* what = argv[3];
      ioFile iocf;
      if (!iocf.check(modelFileName, 1))
        {
          return 1;
        }

      char convert[10];

      NeuralModel* model;
      READMODEL(model, 0, modelFileName);

      ioFile* iof = new ioFile();
      char outputModelFileName[260];
      if (!strcmp(what, "l") || !strcmp(what, "a"))
        {
          strcpy(outputModelFileName, prefixParas);
          strcat(outputModelFileName, "LookupTable");
          iof->takeWriteFile(outputModelFileName);
          model->baseNetwork->lkt->weight.write(iof);
        }
      int i;
      int step = 2;
      if (model->nonLinearType != TANH && model->nonLinearType != SIGM)
        {
          step = 1;
        }
      if (!strcmp(what, "b") || !strcmp(what, "a"))
        {
          for (i = 0; i < model->baseNetwork->size; i += step)
            {
              strcpy(outputModelFileName, prefixParas);
              strcat(outputModelFileName, "BaseWeight");
              sprintf(convert, "%d", i);
              strcat(outputModelFileName, convert);
              iof->takeWriteFile(outputModelFileName);
              model->baseNetwork->modules[i]->weight.write(iof);
              strcpy(outputModelFileName, prefixParas);
              strcat(outputModelFileName, "BaseBias");
              strcat(outputModelFileName, convert);
              iof->takeWriteFile(outputModelFileName);
              model->baseNetwork->modules[i]->bias.write(iof);
            }
        }
      if (!strcmp(what, "o") || !strcmp(what, "a"))
        {
          for (i = 0; i < model->outputNetworkNumber; i++)
            {
              strcpy(outputModelFileName, prefixParas);
              strcat(outputModelFileName, "OutputWeight");
              sprintf(convert, "%d", i);
              strcat(outputModelFileName, convert);
              iof->takeWriteFile(outputModelFileName);
              model->outputNetwork[i]->weight.write(iof);
              strcpy(outputModelFileName, prefixParas);
              strcat(outputModelFileName, "OutputBias");
              strcat(outputModelFileName, convert);
              iof->takeWriteFile(outputModelFileName);
              model->outputNetwork[i]->bias.write(iof);
            }
        }
      if (!strcmp(what, "c") || !strcmp(what, "a"))
        {
          strcpy(outputModelFileName, prefixParas);
          strcat(outputModelFileName, "outputNetworkSize");
          iof->takeWriteFile(outputModelFileName);
          model->outputNetworkSize.write(iof);
          strcpy(outputModelFileName, prefixParas);
          strcat(outputModelFileName, "codeWord");
          iof->takeWriteFile(outputModelFileName);
          model->codeWord.write(iof);
        }
      if (!strcmp(what, "v") || !strcmp(what, "a"))
        {
          strcpy(outputModelFileName, prefixParas);
          strcat(outputModelFileName, "inputVoc");
          iof->takeWriteFile(outputModelFileName);
          int i;
          VocNode* run;
          for (i = 0; i < model->inputVoc->tableSize; i++)
            {
              run = model->inputVoc->table[i];
              while (run->next != NULL)
                {
                  run = run->next;
                  iof->writeNorString(run->word);
                }
            }

          strcpy(outputModelFileName, prefixParas);
          strcat(outputModelFileName, "outputVoc");
          iof->takeWriteFile(outputModelFileName);
          for (i = 0; i < model->outputVoc->tableSize; i++)
            {
              run = model->outputVoc->table[i];
              while (run->next != NULL)
                {
                  run = run->next;
                  iof->writeNorString(run->word);
                }
            }
        }

      delete iof;
      delete model;
    }
  return 0;

}

