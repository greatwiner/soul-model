#include "mainModel.H"

int
main(int argc, char *argv[])
{
  if (argc != 5)
    {
      cout << "modelFileName blockSize textFileName textType" << endl;
      cout
          << "textType: n:normal(text), l:list of ngram (words), id:list of ngram (ids)"
          << endl;
      return 0;
    }
  else
    {
      time_t start, end;
      char* modelFileName = argv[1];
      int blockSize = atoi(argv[2]);
      char* textFileName = argv[3];
      string textType = argv[4];
      if (textType != "n" && textType != "l" && textType != "id" && textType
          != "r")
        {
          cerr << "What is textType?" << endl;
          return 0;
        }

      ioFile iofC;
      if (!iofC.check(modelFileName, 1))
        {
          return 1;
        }
      if (!iofC.check(textFileName, 1))
        {
          return 1;
        }

      NeuralModel* model;
      READMODEL(model, blockSize, modelFileName);

      time(&start);
      float perplexity = model->computePerplexity(model->dataSet, textFileName, textType);
      int countNgram;
      countNgram = model->dataSet->ngramNumber;
      time(&end);
      cout << "With model " << modelFileName << ", perplexity of "
          << textFileName << " is " << perplexity << " (" << countNgram
          << " ngrams)" << endl;
      cout << "Finish after " << difftime(end, start) / 60 << " minutes"
          << endl;
      delete model;
    }
  return 0;
}

