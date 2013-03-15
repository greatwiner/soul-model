#include "mainModel.H"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>

#include <ext/hash_map>

using namespace __gnu_cxx;

int
random(realTensor& tensor_probs, boost::mt19937& gen)
{
  std::vector<double> probs;
  int i;
  for (i = 0; i < tensor_probs.length; i++)
    {
      probs.push_back(tensor_probs.data[i]);
    }
  boost::random::discrete_distribution<> dist(probs.begin(), probs.end());
  return dist(gen);
}

int
main(int argc, char *argv[])
{

  if (argc != 5)
    {
      cout << "modelFileName cont sentenceNumber outputFileName" << endl;
      cout << "cont = 0 for normal n-gram models" << endl;
      cout << "cont = 1 for models using word in previous sentences" << endl;
      return 0;
    }
  else
    {
      time_t start, end;
      char* modelFileName = argv[1];
      int cont = atoi(argv[2]);
      int nsent = atoi(argv[3]);
      char* outputFileName = argv[4];
      ioFile iofC;
      if (!iofC.check(modelFileName, 1))
        {
          return 1;
        }
      if (iofC.check(outputFileName, 0))
        {
          cerr << "output file exists" << endl;
          return 1;
        }

      ioFile iof;
      iof.format = TEXT;
      iof.takeWriteFile(outputFileName);

      NeuralModel* model;
      READMODEL(model, 1, modelFileName);

      time(&start);

      int i, j;

      // Build a tree from word code
      vector<hash_map<int, int> > tree;
      for (i = 0; i < model->outputNetworkNumber; i++)
        {
          hash_map<int, int> branch;
          tree.push_back(branch);
        }

      for (i = 0; i < model->codeWord.size[0]; i++)
        {

          for (j = 0; j < model->codeWord.size[1]; j += 2)
            {
              if (model->codeWord(i, j) == -1)
                {
                  tree[model->codeWord(i, j - 2)][model->codeWord(i, j - 1)]
                      = -1 - i;
                  break;
                }
              else if (j == model->codeWord.size[1] - 2)
                {
                  tree[model->codeWord(i, j)][model->codeWord(i, j + 1)] = -1
                      - i;
                }
              else
                {
                  tree[model->codeWord(i, j)][model->codeWord(i, j + 1)]
                      = model->codeWord(i, j + 2);
                }
            }
          if (model->codeWord(i, model->codeWord.size[1] - 2) != -1)
            {

            }
        }
      // I don't have index to word, so create it
      hash_map<int, string> mapVoc;

      VocNode* run;
      for (i = 0; i < model->inputVoc->tableSize; i++)
        {
          run = model->inputVoc->table[i];
          while (run->next != NULL)
            {
              run = run->next;
              mapVoc[run->index] = run->word;
            }
        }

      // Random generator
      boost::mt19937 gen((unsigned int) time(NULL) + getpid());

      intTensor context;
      context.resize(model->n - 1, 1);
      context = model->inputVoc->ss;
      // inputVoc = outputVoc for now
      int id;
      int innode;
      int outnode;
      realTensor buff;
      int rsent;
      rsent = 0;
      int wcount;

      while (rsent < nsent) // Iteration for sentence
        {
          if (rsent % 100000 == 0)
            {
              cout << rsent << " ... " << flush;
            }
          id = model->inputVoc->ss;
          if (!cont) // Start with <s> ... <s>
            {
              context = model->inputVoc->ss;
            }
          wcount = 0;
          // Generate words until having </s>
          while (id != model->inputVoc->es)
            {
              model->baseNetwork->forward(context);
              buff = model->outputNetwork[0]->forward(model->contextFeature);
              innode = 0;
              outnode = random(buff, gen);

              while (true)
                {
                  innode = tree[innode][outnode];
                  if (innode < 0)
                    {
                      id = -1 - innode;
                      break;
                    }
                  else
                    {
                      buff = model->outputNetwork[innode]->forward(
                          model->contextFeature);
                      outnode = random(buff, gen);
                    }
                }
              wcount = wcount + 1;
              for (i = 0; i < model->n - 2; i++)
                {
                  context(i) = context(i + 1);
                }
              context(model->n - 2) = id;

              if (id != model->inputVoc->es)
                {
                  *(iof.fo) << mapVoc[id] << " ";
                }
              else //if (wcount > 1) // Don't write any empty sentence
                {
                  *(iof.fo) << endl;
                  rsent++;
                }
            }
        }
      cout << endl;
      iof.freeWriteFile();
      time(&end);
      cout << "Finish after " << difftime(end, start) / 60 << " minutes"
          << endl;
      //delete model;
    }
  return 0;

}

