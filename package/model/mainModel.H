#ifndef MODEL_INC
#define MODEL_INC

#include "mainModule.H"
#include "text.H"
#include "NeuralModel.H"
#include "NgramModel.H"
//#include "NgramModel_Bayes.H"
#include "NgramRankModel.H"
#include "NgramWordTranslationModel.H"
#include "NgramPhraseTranslationModel.H"
#include "RecurrentModel.H"
#include "FunctionModel.H"
#include "MultiplesNeuralModel.H"
#include "JointNgramWordTranslationModel.H"

#define READMODEL(sourceModel, blockSize, sourceModelFileName)\
  ioFile iof##sourceModel;\
  string name##sourceModel = iof##sourceModel.recognition(sourceModelFileName);\
  cout << name##sourceModel << endl;\
  if (name##sourceModel == OVR || name##sourceModel == COVR)\
  {\
    sourceModel = new RecurrentModel();\
  }\
  else if (name##sourceModel == WTOVN || name##sourceModel == WTOVN_NCE)\
  {\
	/*cout << "mainModel::READMODEL here" << endl;*/\
    sourceModel = new NgramWordTranslationModel();\
  }\
  else if (name##sourceModel == PTOVN )\
  {\
    sourceModel = new NgramPhraseTranslationModel();\
  }\
  else if (name##sourceModel == RANKOVN || name##sourceModel == RANKCN)\
  {\
    sourceModel = new NgramRankModel();\
  }\
  else\
  {\
    sourceModel = new NgramModel();\
  }\
  iof##sourceModel.takeReadFile(sourceModelFileName);\
  sourceModel->read(&iof##sourceModel, 1, blockSize);

#define READMODEL_MULTIPLE(sourceModel, blockSize, sourceModelFileName)\
  ioFile iof##sourceModel;\
  string name##sourceModel = iof##sourceModel.recognition(sourceModelFileName);\
  if (name##sourceModel == JWTOVN)\
  {\
    sourceModel = new JointNgramWordTranslationModel();\
  }\
  iof##sourceModel.takeReadFile(sourceModelFileName);\
  sourceModel->read(&iof##sourceModel, 1, blockSize);
#endif
