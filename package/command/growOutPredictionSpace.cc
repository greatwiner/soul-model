/*
 Grow , shortlist doesn't include <s>, </s>, <UNK>
 outputVoc = inputVoc except words in shortlist, we add prefix PREFIX_OUT in ioFile.H
 */
#include "mainModel.H"
int
main(int argc, char *argv[])
{
  if (argc != 4)
    {
      cout << "shortlistModelFileName oShortlistModelFileName outModelFileName"
          << endl;
      return 0;
    }
  else
    {
      string prefix = PREFIX_OUT;
      srand48(time(NULL));
      srand(time(NULL));
      char* slModelFileName = argv[1];
      char* oslModelFileName = argv[2];
      char* outModelFileName = argv[3];
      ioFile iofC;
      if (!iofC.check(slModelFileName, 1))
        {
          return 1;
        }
      if (!iofC.check(oslModelFileName, 1))
        {
          return 1;
        }
      if (iofC.check(outModelFileName, 0))
        {
          cerr << "Output file exists" << endl;
          return 1;
        }

      NeuralModel* slModel;
      READMODEL(slModel, 0, slModelFileName);
      NeuralModel* oslModel;
      READMODEL(oslModel, 0, oslModelFileName);

      //New outputNetwork, + number of words in shortlist in voc, + <UNK> + <s> + </s>
      //First, count number of words in shortlist in voc
      VocNode* run;
      int newIndex;
      int nWordsInVoc = 0;
      int i;
      for (i = 0; i < slModel->outputVoc->tableSize; i++)
        {
          run = slModel->outputVoc->table[i];
          while (run->next != NULL)
            {
              run = run->next;
              newIndex = oslModel->outputVoc->index(prefix + run->word);
              if (newIndex != ID_UNK)
                {
                  nWordsInVoc++;
                }
            }
        }

      oslModel->mapOUnk = 1;
      int newSize0 = oslModel->outputNetworkSize(0) + nWordsInVoc + 3;
      LinearSoftmax* newSl = new LinearSoftmax(oslModel->hiddenLayerSize,
          newSize0, oslModel->blockSize, oslModel->otl);
      //Copy outputNetwork[0]
      floatTensor selectOWeight;
      floatTensor selectNWeight;
      // Copy weights of oos words
      selectNWeight.sub(newSl->weight, 0, oslModel->hiddenLayerSize - 1, 0,
          oslModel->outputNetworkSize(0) - 1);
      selectNWeight.copy(oslModel->outputNetwork[0]->weight);
      selectNWeight.sub(newSl->bias, 0, oslModel->outputNetworkSize(0) - 1, 0,
          0);
      selectNWeight.copy(oslModel->outputNetwork[0]->bias);

      // Copy weights of shortlist words
      int idSl = oslModel->outputNetworkSize(0);
      for (i = 0; i < slModel->outputVoc->tableSize; i++)
        {
          run = slModel->outputVoc->table[i];
          while (run->next != NULL)
            {
              run = run->next;
              newIndex = oslModel->outputVoc->index(prefix + run->word);
              if (newIndex != ID_UNK)
                {
                  oslModel->codeWord(newIndex, 0) = 0;
                  oslModel->codeWord(newIndex, 1) = idSl;
                  selectOWeight.select(slModel->outputNetwork[0]->weight, 1,
                      run->index);
                  selectNWeight.select(newSl->weight, 1, idSl);
                  selectNWeight.copy(selectOWeight);
                  newSl->bias(idSl) = slModel->outputNetwork[0]->bias(
                      run->index);
                  idSl++;
                }
            }
        }
      newIndex = oslModel->outputVoc->index(prefix + SS);
      oslModel->codeWord(newIndex, 0) = 0;
      oslModel->codeWord(newIndex, 1) = idSl;
      idSl++;
      newIndex = oslModel->outputVoc->index(prefix + ES);
      oslModel->codeWord(newIndex, 0) = 0;
      oslModel->codeWord(newIndex, 1) = idSl;
      idSl++;
      newIndex = oslModel->outputVoc->index(prefix + UNK);
      oslModel->codeWord(newIndex, 0) = 0;
      oslModel->codeWord(newIndex, 1) = idSl;
      idSl++;

      oslModel->outputNetworkSize(0) = newSize0;
      delete oslModel->outputNetwork[0];
      oslModel->outputNetwork[0] = newSl;
      SoulVocab* oldVoc = oslModel->outputVoc;
      oslModel->outputVoc = oslModel->inputVoc;
      ioFile oIof;
      oIof.takeWriteFile(outModelFileName);
      oslModel->write(&oIof, 1);
      oslModel->outputVoc = oldVoc;
      delete slModel;
      delete oslModel;
    }
  return 0;
}

