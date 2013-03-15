class VocNode
{
public:
  VocNode();
  VocNode(string word, int idWord);
  ~VocNode();
  string word;
  int index;
  VocNode* next;
};

class SoulVocab
{
public:
  int tableSize;
  VocNode **table;
  VocNode **runTable;
  int wordNumber;
  SoulVocab();
  SoulVocab(char* dataFileName, char* indexFileName);
  SoulVocab(char* dataFileName);
  SoulVocab(SoulVocab* inVocab);
  void
  read(ioFile* iof);

  ~SoulVocab();
  int
  getHashValue(string word);
  int
  add(string word, int idWord); // Add one word
  int
  add(string word, int idWord, int hashValue); // Add one word
  int
  index(string word);
  void
  write(ioFile* iof);

  int ss;
  int es;
  int unk;
};
