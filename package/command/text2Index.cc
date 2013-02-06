#include "mainModel.H"

int checkBlankString(string line)
{
  for (int i = 0; i < line.length(); i++)
    {
      if (line[i] != ' ')
        {
          return 0;
        }
    }
  return 1;
}
int
main(int argc, char *argv[])
{
  if (argc != 4)
    {
      cout << "textFileName vocFileName indexFileName" << endl;
      return 0;
    }
  else
    {
      char* textFileName = argv[1];
      char* vocFileName = argv[2];
      char* indexFileName = argv[3];
      ioFile iofC;
      if (!iofC.check(textFileName, 1))
        {
          return 1;
        }
      if (!iofC.check(vocFileName, 1))
        {
          return 1;
        }
      if (iofC.check(indexFileName, 0))
        {
          cerr << "index file exists" << endl;
          return 1;
        }
      string line;
      int id;
      int readLineNumber = 0;
      SoulVocab* voc = new SoulVocab(vocFileName);
      cout << "write to file:" << indexFileName << endl;
      ioFile iof;
      iof.format = TEXT;
      iof.takeReadFile(textFileName);
      iof.takeWriteFile(indexFileName);
      while (!iof.getEOF())
      {
         if (iof.getLine(line))
           {
             if(!checkBlankString(line))
               {
               istringstream streamLine(line + " </s>");
               string word;
               while (streamLine >> word)
                 {
                 id = voc->index(word);
                 iof.writeInt(id);  
                 }
               }
           }
         readLineNumber++;
#if PRINT_DEBUG
         if (readLineNumber % NLINEPRINT == 0)
           {
             cout << readLineNumber << " ... " << flush;
           }
#endif
       }
#if PRINT_DEBUG
      cout << endl;
#endif
      delete voc;
    }
  return 0;
}

