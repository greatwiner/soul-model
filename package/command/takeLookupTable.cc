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

      NeuralModel* model;
      READMODEL(model, 0, modelFileName);

      ioFile* iof = new ioFile();
      char inputFileName[260];
      strcpy(inputFileName, prefixParas);
      strcat(inputFileName, "LookupTable");
      if (!iocf.check(inputFileName, 1))
        {
          return 1;
        }
      iof->takeReadFile(inputFileName);
      model->baseNetwork->lkt->weight.read(iof);

      ioFile oIof;
      oIof.takeWriteFile(outputModelFileName);
      model->write(&oIof);

      delete iof;
      delete model;
    }
  return 0;
}

