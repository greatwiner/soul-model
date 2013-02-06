#include "mainModel.H"

int
main(int argc, char *argv[])
{
  if (argc != 5)
    {
      cout << "modelFileName blockSize textFileName outputFileName"
          << endl;
      return 0;
    }
  else
    {
      time_t start, end;
      char* modelFileName = argv[1];
      int blockSize = atoi(argv[2]);
      char* textFileName = argv[3];
      char* outputFileName = argv[4];
      ioFile iofC;
      if (!iofC.check(modelFileName, 1))
        {
          return 1;
        }
      if (!iofC.check(textFileName, 1))
        {
          return 1;
        }
      if (iofC.check(outputFileName, 0))
        {
          cerr << "prob file exists" << endl;
          return 1;
        }

      NeuralModel* model;
      READMODEL(model, blockSize, modelFileName);
      time(&start);

      ioFile iof;
      iof.format = TEXT;
      iof.takeReadFile((char*) textFileName);
      ioFile iofO;
      iofO.format = TEXT;
      iofO.takeWriteFile(outputFileName);
      int readLineNumber = 0;

      char * maxNgramNumberEnv;
      int maxNgramNumber = BLOCK_NGRAM_NUMBER;
      maxNgramNumberEnv = getenv("BLOCK_NGRAM_NUMBER");
      if (maxNgramNumberEnv != NULL)
        {
          maxNgramNumber = atoi(maxNgramNumberEnv);
        }

      while (!iof.getEOF())
        {
          model->dataSet->addLine(&iof);
          readLineNumber++;
#if PRINT_DEBUG
          if (readLineNumber % NLINEPRINT == 0 && readLineNumber != 0)
            {
              cout << readLineNumber << " ... " << flush;
            }
#endif
          if (model->dataSet->ngramNumber > maxNgramNumber
              - MAX_WORD_PER_SENTENCE)
            {
              cout << "Compute " << model->dataSet->ngramNumber << " ngrams"
                  << endl;
              model->dataSet->createTensor();
              model->forwardProbability(model->dataSet->dataTensor,
                  model->dataSet->probTensor);
              for (int i = 0; i < model->dataSet->probTensor.length; i++)
                {
                  iofO.writeFloat(model->dataSet->probTensor(i));
                }
              model->dataSet->reset();
            }
        }
      if (model->dataSet->ngramNumber != 0)
        {
          cout << "Compute with " << model->dataSet->ngramNumber << " ngrams"
              << endl;
          model->dataSet->createTensor();
          model->forwardProbability(model->dataSet->dataTensor,
              model->dataSet->probTensor);
          for (int i = 0; i < model->dataSet->probTensor.length; i++)
            {
              iofO.writeFloat(model->dataSet->probTensor(i));
            }
        }

#if PRINT_DEBUG
      cout << endl;
#endif
      time(&end);
      cout << "Finish after " << difftime(end, start) / 60 << " minutes"
          << endl;
      delete model;
    }
  return 0;
}

