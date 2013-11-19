#include "mainModel.H"

int
main(int argc, char *argv[]) {
	ioFile modelFile_en;
	ioFile modelFile_fr;
	modelFile_en.takeReadFile(argv[1]);
	modelFile_fr.takeReadFile(argv[2]);
	NeuralModel* model_en = new NgramModel();
	NeuralModel* model_fr = new NgramModel();
	int blockSize = 128;
	model_en->read(&modelFile_en, 1, blockSize);
	model_fr->read(&modelFile_fr, 1, blockSize);
	int index_en1 = model_en->inputVoc->index("france");
	floatTensor temp;
	floatTensor embedding_en1;
	temp.select(model_en->baseNetwork->lkt->weight, 1, index_en1);
	embedding_en1.copy(temp);
	// for test
	cout << "similarity::main finish read embedding_en1: " << index_en1 << endl;
	int index_en2 = model_en->inputVoc->index("paris");
	floatTensor embedding_en2;
	temp.select(model_en->baseNetwork->lkt->weight, 1, index_en2);
	embedding_en2.copy(temp);
	// for test
	cout << "similarity::main finish read embedding_en2: " << index_en2 << endl;
	int index_fr1 = model_en->inputVoc->index("japan");
	floatTensor embedding_fr1;
	temp.select(model_en->baseNetwork->lkt->weight, 1, index_fr1);
	embedding_fr1.copy(temp);
	// for test
	cout << "similarity::main finish read embedding_fr1: " << index_fr1 << endl;
	floatTensor embedding_fr2;
	float scal = sqrt(embedding_fr1.averageSquare()/embedding_en1.averageSquare());
	// for test
	cout << "similarity::main scal: " << scal << endl;

	embedding_fr2.copy(embedding_fr1);
	embedding_fr2.axpy(embedding_en2, 1);
	embedding_fr2.axpy(embedding_en1, -1);

	floatTensor dist(model_en->inputVoc->wordNumber, 1);
	int indexMax;
	float max = -1000;
	for (int i = 0; i < model_en->inputVoc->wordNumber; i ++) {
		temp.select(model_en->baseNetwork->lkt->weight, 1, i);
		//floatTensor temp1;
		//temp1.copy(temp);
		//temp1.axpy(embedding_fr2, -1);
		dist(i, 0) = temp.angleDist(embedding_fr2);
		//dist(i, 0) = temp1.averageSquare();
		if ((dist(i, 0) > max || i == 0)) {
			indexMax = i;
			max = dist(i, 0);
		}
	}
	// for test
	cout << "similarity::main indexMax: " << indexMax << endl;
	VocNode* run;
	for (int j = 0; j < model_en->inputVoc->tableSize; j++) {
		run = model_en->inputVoc->table[j];
		while (run->next != NULL) {
			run = run->next;
			if (run->index == indexMax) {
				cout << run->word << endl;
				cout << max << endl;
				break;
				break;
			}
		}
	}
}
