#include "mainModel.H"
#include "time.h"
int
main(int argc, char *argv[])
{
  // Training paras
  int nOutputLayer = 2;
  string nonLinearType = SIGM;
  int projectionDimension = 4;
  int hiddenLayerSize = 4;
  int maxExampleNumber = 12000;
  float weightDecay = 3e-05;

  string learningRateType = "n";
  float learningRate = 0.1;
  float learningRateDecay = 0.00001;

  learningRateType = "d";
  learningRate = 0.1;
  learningRateDecay = 2;

  string name;
  //Un modified paras
  int blockSize = 4;
  int newBlockSize = 2;
  int outputSize = 4;
  int inputSize = 4;
  int n = 3;

  intTensor codeWord;
  intTensor outputNetworkSize;
  intTensor context(n - 1, blockSize);
  intTensor word(blockSize, 1);
  floatTensor coefTensor(blockSize, 1);
  floatTensor coefTensor2(2, 1);

  // take all coefficients 1 for each n-gram, can change here
  coefTensor = 1;
  coefTensor2 = 1;

  context(0, 0) = 0;
  context(1, 0) = 1;
  context(0, 1) = 1;
  context(1, 1) = 2;
  context(0, 2) = 2;
  context(1, 2) = 3;
  context(0, 3) = 3;
  context(1, 3) = 0;
  word(0) = 2;
  word(1) = 3;
  word(2) = 0;
  word(3) = 1;

  intTensor context2(n - 1, 2);
  intTensor word2(2, 1);
  context2(0, 0) = 0;
  context2(1, 0) = 1;
  context2(0, 1) = 1;
  context2(1, 1) = 2;
  word2(0) = 2;
  word2(1) = 3;

  intTensor context3(n - 1, 2);
  intTensor word3(2, 1);
  context3(0, 0) = 2;
  context3(1, 0) = 3;
  context3(0, 1) = 3;
  context3(1, 1) = 0;
  word3(0) = 0;
  word3(1) = 1;

  if (nOutputLayer == 2)
    {

      codeWord.resize(4, 4);
      codeWord(0, 0) = 0;
      codeWord(0, 1) = 0;
      codeWord(0, 2) = 1;
      codeWord(0, 3) = 0;
      codeWord(1, 0) = 0;
      codeWord(1, 1) = 0;
      codeWord(1, 2) = 1;
      codeWord(1, 3) = 1;
      codeWord(2, 0) = 0;
      codeWord(2, 1) = 1;
      codeWord(2, 2) = 2;
      codeWord(2, 3) = 0;
      codeWord(3, 0) = 0;
      codeWord(3, 1) = 1;
      codeWord(3, 2) = 2;
      codeWord(3, 3) = 1;

      outputNetworkSize.resize(3, 1);
      outputNetworkSize = 2;
    }
  else if (nOutputLayer == 1)
    {
      codeWord.resize(4, 2);
      codeWord(0, 0) = 0;
      codeWord(0, 1) = 0;
      codeWord(1, 0) = 0;
      codeWord(1, 1) = 1;
      codeWord(2, 0) = 0;
      codeWord(2, 1) = 2;
      codeWord(3, 0) = 0;
      codeWord(3, 1) = 3;

      outputNetworkSize.resize(1, 1);
      outputNetworkSize = 4;
    }
  intTensor hiddenLayerSizeArray;
  hiddenLayerSizeArray.resize(1, 1);
  hiddenLayerSizeArray = hiddenLayerSize;
  NeuralModel* model;
  //Run

  name = CN;
  cout << "Classical Ngram Model" << endl;
  model = new NgramModel(name, inputSize, outputSize, blockSize, n,
      projectionDimension, nonLinearType, hiddenLayerSizeArray, codeWord,
      outputNetworkSize);

  cout << "prior probs:" << endl;
  model->forwardOne(context, word);

  model->probabilityOne.write();
  model->trainTest(maxExampleNumber, weightDecay, learningRateType,
      learningRate, learningRateDecay, context, word, coefTensor);
  cout << "posterior probs:" << endl;
  model->forwardOne(context, word);
  model->probabilityOne.write();

  delete model;

  name = OVN;
  cout << "One Vector Ngram Model" << endl;
  model = new NgramModel(name, inputSize, outputSize, blockSize, n,
      projectionDimension, nonLinearType, hiddenLayerSizeArray, codeWord,
      outputNetworkSize);

  cout << "prior probs:" << endl;
  model->forwardOne(context, word);
  model->probabilityOne.write();
  model->trainTest(maxExampleNumber, weightDecay, learningRateType,
      learningRate, learningRateDecay, context, word, coefTensor);
  cout << "posterior probs:" << endl;
  model->forwardOne(context, word);
  model->probabilityOne.write();

  delete model;

  name = ROVN;
  cout << "Recurrent One Vector Ngram Model" << endl;
  model = new NgramModel(name, inputSize, outputSize, blockSize, n,
      projectionDimension, nonLinearType, hiddenLayerSizeArray, codeWord,
      outputNetworkSize);

  cout << "prior probs:" << endl;
  model->forwardOne(context, word);

  model->probabilityOne.write();
  model->trainTest(maxExampleNumber, weightDecay, learningRateType,
      learningRate, learningRateDecay, context, word, coefTensor);
  cout << "posterior probs:" << endl;
  model->forwardOne(context, word);
  model->probabilityOne.write();
  delete model;

  name = MAXOVN;
  cout << "Max One Vector Ngram Model" << endl;
  model = new NgramModel(name, inputSize, outputSize, blockSize, n,
      projectionDimension, nonLinearType, hiddenLayerSizeArray, codeWord,
      outputNetworkSize);

  cout << "prior probs:" << endl;
  model->forwardOne(context, word);

  model->probabilityOne.write();
  model->trainTest(maxExampleNumber, weightDecay, learningRateType,
      learningRate, learningRateDecay, context, word, coefTensor);
  cout << "posterior probs:" << endl;
  model->forwardOne(context, word);
  model->probabilityOne.write();

  delete model;

  return 0;
}

