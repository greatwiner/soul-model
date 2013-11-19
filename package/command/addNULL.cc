#include "mainModel.H"

int
main(int argc, char *argv[])
{
  if (argc != 3)
    {
      cout << "modelFileName outputModelFileName" << endl;
      return 0;
    }
  char* modelFileName = argv[1];
  char* outputModelFileName = argv[2];
  ioFile iofC;
  if (!iofC.check(modelFileName, 1))
    {
      return 1;
    }
  if (iofC.check(outputModelFileName, 0))
    {
      cerr << "Prototype exists" << endl;
      return 1;
    }

  NeuralModel* model;
  int i;
  READMODEL(model, 1, modelFileName);
  int NULLindex = model->inputVoc->index("NULL");
  // for test
  cout << "addNULL::main NULLindex: " << NULLindex << endl;
  if (NULLindex == ID_UNK)
    {
      model->inputVoc->add("NULL", model->inputVoc->wordNumber);
      floatTensor newWeight;
      floatTensor subW;
      
      newWeight.resize(model->baseNetwork->lkt->weight.size[0],
          model->baseNetwork->lkt->weight.size[1] + 1);
      newWeight = 0;
      subW.sub(newWeight, 0, model->baseNetwork->lkt->weight.size[0] - 1, 0,
          model->baseNetwork->lkt->weight.size[1] - 1);
      subW.copy(model->baseNetwork->lkt->weight);

      delete[] model->baseNetwork->lkt->weight.data;

      model->baseNetwork->lkt->weight.data = newWeight.data;
      newWeight.haveMemory = 0;
      model->baseNetwork->lkt->weight.size[1]++;
      model->baseNetwork->lkt->weight.length
          += model->baseNetwork->lkt->weight.size[0];
    }
  intTensor newCodeWord;
  NULLindex = model->outputVoc->index("NULL");
  // for test
  cout << "addNULL::main NULLindex: " << NULLindex << endl;
  if (NULLindex == ID_UNK)
    {
      model->outputVoc->add("NULL", model->outputVoc->wordNumber);
      newCodeWord.resize(model->outputVoc->wordNumber, model->codeWord.size[1]);
      int j;
      for (i = 0; i < model->outputVoc->wordNumber - 1; i ++)
      {
         for (j = 0; j < model->codeWord.size[1]; j ++)
         {
            newCodeWord(i, j) = model->codeWord(i, j);
         }
      }
      NULLindex = model->outputVoc->wordNumber - 1;
      newCodeWord(NULLindex, 0) = 0;
      newCodeWord(NULLindex, 1) = model->outputNetwork[0]->weight.size[1];
      for (i = 2; i < model->codeWord.size[1]; i++)
        {
          newCodeWord(NULLindex, i) = -1;
        }


      model->codeWord.length += model->codeWord.size[1];
      model->codeWord.size[0]++;
      model->codeWord.stride[1] = model->codeWord.size[0];
      delete [] model->codeWord.data;
      model->codeWord.data = newCodeWord.data;
      newCodeWord.haveMemory = 0;

      floatTensor newWeight;
      floatTensor newBias;
      newWeight.resize(model->outputNetwork[0]->weight.size[0], model->outputNetwork[0]->weight.size[1] + 1);
      newWeight = 0;
      floatTensor subWeight;
      subWeight.sub(newWeight, 0, model->outputNetwork[0]->weight.size[0] - 1, 0, model->outputNetwork[0]->weight.size[1] - 1);      
      subWeight.copy(model->outputNetwork[0]->weight);
      delete[] model->outputNetwork[0]->weight.data;
      model->outputNetwork[0]->weight.data = newWeight.data;
      newWeight.haveMemory = 0;


 
      newBias.resize(model->outputNetwork[0]->bias.size[0], model->outputNetwork[0]->bias.size[1] + 1);
      newBias = 0;
      floatTensor subBias;
      subBias.sub(newBias, 0, model->outputNetwork[0]->bias.size[0] - 1, 0, model->outputNetwork[0]->bias.size[1] - 1);      
      subBias.copy(model->outputNetwork[0]->bias);
      delete[] model->outputNetwork[0]->bias.data;
      model->outputNetwork[0]->bias.data = newBias.data;
      newBias.haveMemory = 0;
 


      model->outputNetwork[0]->weight.size[1]++;
      model->outputNetwork[0]->weight.length
          = model->outputNetwork[0]->weight.size[0]
              * model->outputNetwork[0]->weight.size[1];
      model->outputNetworkSize(0)++;
      model->outputNetwork[0]->bias.size[0]++;
      model->outputNetwork[0]->bias.length++;
    }
  else if (model->codeWord(NULLindex, 2) != -1)
    {
      model->codeWord(NULLindex, 0) = 0;
      model->codeWord(NULLindex, 1) = model->outputNetwork[0]->weight.size[1];
      for (i = 2; i < model->codeWord.size[1]; i++)
        {
          model->codeWord(NULLindex, i) = -1;
        }

      floatTensor newWeight;
      floatTensor newBias;
      newWeight.resize(model->outputNetwork[0]->weight.size[0], model->outputNetwork[0]->weight.size[1] + 1);
      newWeight = 0;
      floatTensor subWeight;
      subWeight.sub(newWeight, 0, model->outputNetwork[0]->weight.size[0] - 1, 0, model->outputNetwork[0]->weight.size[1] - 1);      
      subWeight.copy(model->outputNetwork[0]->weight);
      delete[] model->outputNetwork[0]->weight.data;
      model->outputNetwork[0]->weight.data = newWeight.data;
      newWeight.haveMemory = 0;


 
      newBias.resize(model->outputNetwork[0]->bias.size[0], model->outputNetwork[0]->bias.size[1] + 1);
      newBias = 0;
      floatTensor subBias;
      subBias.sub(newBias, 0, model->outputNetwork[0]->bias.size[0] - 1, 0, model->outputNetwork[0]->bias.size[1] - 1);      
      subBias.copy(model->outputNetwork[0]->bias);
      delete[] model->outputNetwork[0]->bias.data;
      model->outputNetwork[0]->bias.data = newBias.data;
      newBias.haveMemory = 0;

      model->outputNetwork[0]->weight.size[1]++;
      model->outputNetwork[0]->weight.length
          = model->outputNetwork[0]->weight.size[0]
              * model->outputNetwork[0]->weight.size[1];
      model->outputNetworkSize(0)++;
      model->outputNetwork[0]->bias.size[0]++;
      model->outputNetwork[0]->bias.length++;
    }
  ioFile iof;
  iof.takeWriteFile(outputModelFileName);
  model->write(&iof, 1);
  delete model;
}

