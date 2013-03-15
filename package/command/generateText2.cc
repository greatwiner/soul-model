#include "mainModel.H"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/discrete_distribution.hpp>

#include <ext/hash_map>

using namespace __gnu_cxx;

int
random(floatTensor& tensor_probs, boost::mt19937& gen)
{
  std::vector<double> probs;
  int i;
  for (i = 0; i < tensor_probs.length; i++)
    {
      probs.push_back(tensor_probs(i));
    }
  boost::random::discrete_distribution<> dist(probs.begin(), probs.end());
  return dist(gen);
}

int
main(int argc, char *argv[])
{

  if (argc != 5)
    {
      cout << "modelFileName cont sentNumber outputFileName" << endl;
      cout << "cont = 0 for normal n-gram models," << endl;
      cout << "the results are in outputFileName" << endl;
      cout << "cont = 1 for models using word in previous sentences," << endl;
      cout << "the results are in outputFileName{0,1,...}" << endl;
      return 0;
    }
  else
    {
      int blockSize = 128;
      time_t start, end;
      char* modelFileName = argv[1];
      int cont = atoi(argv[2]);
      int sentNumber = atoi(argv[3]);
      char* outputFileName = argv[4];
      ioFile iofC;
      if (!iofC.check(modelFileName, 1))
        {
          return 1;
        }
      if (!cont && iofC.check(outputFileName, 0))
        {
          cerr << "output file exists" << endl;
          return 1;
        }
      int rBlockSize;
      ioFile iof;
      ioFile* aiof;
      char outputModelFileName[260];
      char convert[260];
      char outname[260];

      if (!cont)
        {
          iof.format = TEXT;
          iof.takeWriteFile(outputFileName);
        }
      else
        {
          aiof = new ioFile[blockSize];
          for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++)
            {
              aiof[rBlockSize].format = TEXT;
              strcpy(outname, outputFileName);
              sprintf(convert, "%d", rBlockSize);
              strcat(outname, convert);
              aiof[rBlockSize].takeWriteFile(outname);
            }
        }

      NeuralModel* model;
      READMODEL(model, blockSize, modelFileName);

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

      // Go, go, go, start with context and word are all <s>
      intTensor context;
      context.resize(model->n - 1, blockSize);
      context = model->inputVoc->ss;
      intTensor word;
      word.resize(blockSize, 1);
      word = model->inputVoc->ss;
      // Output

      vector<vector<string> > sents;
      vector<string> sent;

      for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++)
        {
          sents.push_back(sent);
        }
      // inputVoc = outputVoc for now

      int innode;
      int outnode;
      floatTensor outMainLayer;
      floatTensor selectOutMainLayer;
      floatTensor outOtherLayer;
      int rblock;
      rblock = 0;

      intTensor selectContext;
      floatTensor selectContextFeature;
      int nsent = 0;
      while (true) // Iteration, each time generate a block of words

        {
          if (rblock % 100000 == 0)
            {
              cout << rblock << " ... " << flush;
            }

          if (!cont) // Start with <s> ... <s> if seeing </s>, for each flow

            {
              for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++)
                {
                  if (word(rBlockSize) == model->inputVoc->es)
                    {
                      selectContext.select(context, 1, rBlockSize);
                      selectContext = model->inputVoc->ss;
                    }
                }
            }
          // Forward input to hidden
          model->baseNetwork->forward(context);
          // Forward hidden to main softmax layer
          outMainLayer
              = model->outputNetwork[0]->forward(model->contextFeature);
          // Going down the tree
          for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++)
            {
              innode = 0;
              selectOutMainLayer.select(outMainLayer, 1, rBlockSize);
              outnode = random(selectOutMainLayer, gen);
              selectContextFeature.select(model->contextFeature, 1, rBlockSize);
              while (true)
                {
                  innode = tree[innode][outnode];
                  if (innode < 0)
                    {
                      // Ok, find it
                      word(rBlockSize) = -1 - innode;
                      break;
                    }
                  else
                    {
                      outOtherLayer = model->outputNetwork[innode]->forward(
                          selectContextFeature);
                      outnode = random(outOtherLayer, gen);
                    }
                }
              for (i = 0; i < model->n - 2; i++)
                {
                  context(i, rBlockSize) = context(i + 1, rBlockSize);
                }
              context(model->n - 2, rBlockSize) = word(rBlockSize);
              if (word(rBlockSize) != model->inputVoc->es)
                {
                  sents[rBlockSize].push_back(mapVoc[word(rBlockSize)]);
                }
              else // Finish this sentence, output

                {
                  if (!cont)
                    {
                      for (i = 0; i < sents[rBlockSize].size(); i++)
                        {
                          *(iof.fo) << sents[rBlockSize][i] << " ";
                        }
                      *(iof.fo) << endl;
                    }
                  else
                    {
                      for (i = 0; i < sents[rBlockSize].size(); i++)
                        {
                          *(aiof[rBlockSize].fo) << sents[rBlockSize][i] << " ";
                        }
                      *(aiof[rBlockSize].fo) << endl;
                    }
                  sents[rBlockSize].clear();
                  nsent++;
                  if (nsent >= sentNumber)
                    {
                      break;
                    }
                }
            }
          if (nsent >= sentNumber)
            {
              break;
            }
          rblock++;
        }
      cout << endl;
      if (!cont)
        {
          iof.freeWriteFile();
        }
      else
        {
          for (rBlockSize = 0; rBlockSize < blockSize; rBlockSize++)
            {
              aiof[rBlockSize].freeWriteFile();
            }
          delete[] aiof;
        }
      time(&end);
      cout << "Finish after " << difftime(end, start) / 60 << " minutes"
          << endl;
      delete model;
    }
  return 0;
}


