/*
 * absoluteDist.cc
 *
 *  Created on: Apr 7, 2013
 *      Author: dokhanh
 */




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
	tensor1.axpy(tensor2, -1);
	cout << "Absolute distance between tensors in " << argv[1] << " and " << argv[2] << " : " << tensor1.sumSquared()/(tensor1.size[0]*tensor1.size[1]) << endl;
}
