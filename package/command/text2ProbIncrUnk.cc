#include "mainModel.H"

int
main(int argc, char *argv[])
{
  if (argc != 6)
    {
      cout << "modelFileName blockSize incrUnk textFileName outputFileName"
          << endl;
      cout << "incrUnk: *10^incrUnk for unknown word probs" << endl;
      return 0;
    }
  else
    {
	  // for test
	  cout << "text2ProbIncrUnk.cc::main here3" << endl;
      time_t start, end;
      char* modelFileName = argv[1];
      int blockSize = atoi(argv[2]);
      float incrUnk = pow(10, atof(argv[3]));
      char* textFileName = argv[4];
      char* outputFileName = argv[5];
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
      // for test
      cout << "text2ProbIncrUnk.cc::main here4" << endl;
      READMODEL(model, blockSize, modelFileName);
      // for test
      cout << "text2ProbIncrUnk.cc::main here5" << endl;
      model->incrUnk = incrUnk;
      time(&start);

      ioFile iof;
      iof.format = TEXT;
      iof.takeReadFile((char*) textFileName);
      ioFile iofO;
      iofO.format = TEXT;
      iofO.takeWriteFile(outputFileName);
      // for test
      cout << "text2ProbIncrUnk.cc::main here6" << endl;
      int readLineNumber = 0;

      char * maxNgramNumberEnv;
      int maxNgramNumber = MODEL_NGRAM_NUMBER;
      maxNgramNumberEnv = getenv("MODEL_NGRAM_NUMBER");
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
        	  // for test
        	  cout << "text2ProbIncrUnk.cc::main here7" << endl;
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
          // for test
          cout << "text2ProbIncrUnk.cc::main here" << endl;
          model->dataSet->createTensor();
          // for test
          cout << "text2ProbIncrUnk.cc::main here1" << endl;
          model->forwardProbability(model->dataSet->dataTensor,
              model->dataSet->probTensor);
          // for test
          cout << "text2ProbIncrUnk.cc::main here2" << endl;
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

