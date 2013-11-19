#include "mainTensor.H"
#include "mainModel.H"

int
main(int argc, char *argv[]) {
	/*floatTensor tensor;
	tensor.resize(500, 300000);
	outils* otl = new outils();*/
	/*tensor.resize(10, 10);
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			if (i != 5 || j != 5) {
				tensor(i, j) = 1;
			}
			else {
				tensor(i, j) = 1.0/0;
			}
		}
	}
	tensor.write();
	cout << tensor.testInf() << endl;*/
	/*for (int i = 1; ; i++) {
		cout << "testNann::main iteration " << i << endl;
		tensor.initializeNormal(otl);
		ioFile file;
		char name[260];
		char convertStr[260];
		strcpy(name, "/vol/work/dokhanh/wmt13/esLM/0/10gram/matrixRd");
		sprintf(convertStr, "%d", i);
		strcat(name, convertStr);
		file.takeWriteFile(name);
		tensor.write(&file);
		cout << "testNann::main averageSquare: " << tensor.averageSquareBig() << endl;
		int res=tensor.testInf();
		cout << "testNann::main testInf: " << res << endl;
		if (res != 0) {
			exit(0);
		}
	}*/
	char modelFileName[260];
	strcpy(modelFileName, "/vol/work2/dokhanh/wmt13/esLM/allBayes1e6Small2Data/out.per");
	ioFile modelFiles;
	modelFiles.takeReadFile(modelFileName);
	string line;
	while(!modelFiles.getEOF()) {
		modelFiles.getLine(line);
		cout << line << endl;
	}
}
