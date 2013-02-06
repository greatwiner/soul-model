#include "mainModel.H"
int
main(int argc, char *argv[])
{
  if (argc != 2)
    {
      cout << "ngramFileName" << endl;
      return 0;
    }
  char* ngramFileName = argv[1];
  ioFile iofC;
  if (!iofC.check(ngramFileName, 1))
    {
      return 1;
    }
  ioFile* iof = new ioFile();
  iof->format = BINARY;
  iof->takeReadFile(ngramFileName);
  int n;
  int ngramNumber;
  iof->readInt(ngramNumber);
  cout << ngramNumber << " ";
  iof->readInt(n);
  cout << n << endl;

  int *ngram = (int*) malloc(n * sizeof(int));
  if (ngramNumber > 10)
    {
      ngramNumber = 10;
    }
  for (int j = 0; j < ngramNumber; j++)
    {
      iof->readIntArray(ngram, n);
      for (int i = 0; i < n; i++)
        {
          cout << ngram[i] << " ";
        }
      cout << endl;
    }
  free(ngram);
  delete iof;
}

