#include "mainModel.H"
#include "ioFile.H"

int
main(int argc, char *argv[]) {
	if (argc != 3) {
		cout << "fileName1 fileName2" << endl;
		return 1;
	}
	char* fileName1 = argv[1];
	char* fileName2 = argv[2];
	ioFile file;
	file.takeReadFile(fileName1);
	floatTensor tensor1;
	tensor1.read(&file);
	file.takeReadFile(fileName2);
	floatTensor tensor2;
	tensor2.read(&file);
	cout << "Angle distance between tensors in " << argv[1] << " and " << argv[2] << " : " << tensor1.angleDist(tensor2) << endl;
}
