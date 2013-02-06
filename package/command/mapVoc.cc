#include "mainModel.H"

int
main(int argc, char *argv[])
{

  if (argc != 4)
    {
      cout << "vocFileName baseVocFileName mapOutputFileName" << endl;
      return 0;
    }

  char* vocFileName = argv[1];
  char* baseVocFileName = argv[2];
  char* mapOutputFileName = argv[3];

  ioFile iof;
  if (!iof.check(vocFileName, 1))
    {
      return 1;
    }
  if (!iof.check(baseVocFileName, 1))
    {
      return 1;
    }
  if (iof.check(mapOutputFileName, 0))
    {
      cerr << "mapOutput file exists" << endl;
      return 1;
    }

  SoulVocab* baseVoc = new SoulVocab(baseVocFileName);
  iof.format = TEXT;
  iof.takeReadFile(vocFileName);
  iof.takeWriteFile(mapOutputFileName);
  string line;
  while (!iof.getEOF())
  {
     if (iof.getLine(line))
       {
          iof.writeInt(baseVoc->index(line));
       }
  }
  delete baseVoc;
}

