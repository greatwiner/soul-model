NGRAM_PRINT = 1000000
def readPlainTensor(inputFileName, transpose):
	import numpy
	import gzip
	if(inputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'r')
	else:
		f = open(inputFileName, 'r')
	shape = map(int, f.readline().split())
	if(transpose == 'N'):
		data = numpy.zeros(shape, dtype=numpy.float32)
		for i in range(shape[0]):
			data[i, :] = numpy.array(f.readline().split())
	elif(transpose == 'T'):
		data = numpy.zeros((shape[1], shape[0]), dtype=numpy.float32)
		for i in range(shape[0]):
			data[:, i] = numpy.array(f.readline().split())
	else:
		print "read T or N"
	return data

def writePlainTensor(data, outputFileName):
	import numpy
	if(outputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'w')
	else:
		f = open(outputFileName, "w")
	f.write(str(data.shape[0]) + ' ' + str(data.shape[1]) + '\n')
	for i in range(data.shape[0]):
		f.write(' '.join(map(str, data[i,:])) + '\n')
	f.close()
def writePlainIntTensor(data, outputFileName):
	import numpy
	f = open(outputFileName, "w")
	f.write(str(data.shape[0]) + ' ' + str(data.shape[1]) + '\n')
	for i in range(data.shape[0]):
		f.write(' '.join(map(str, data[i,:])) + '\n')
	f.close()


def readTensor(inputFileName, transpose):
	import numpy
	import gzip
	import struct
	if(inputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'rb')
	else:
		f = open(inputFileName, 'rb')
	size = struct.unpack('ii', f.read(8))	
	data = numpy.fromfile(file = f, dtype=numpy.float32).reshape(size, order='F')
	if(transpose == 'N'):
		return data
	elif(transpose == 'T'):
		return data.transpose()
	else:
		print "read T or N?"

def readSCTensor(inputFileName, transpose, y):
	import numpy
	import gzip
	import struct
	if(inputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'rb')
	else:
		f = open(inputFileName, 'rb')
	data = numpy.fromfile(file = f, dtype=numpy.float32)
	size = [0] * 2
	size[1] = y
	size[0] = data.shape[0] / y
	data = data.reshape(size, order='C')
	if(transpose == 'N'):
		return data
	elif(transpose == 'T'):
		return data.transpose()
	else:
		print "read T or N?"


def readCTensor(inputFileName, transpose):
	import numpy
	import gzip
	import struct
	if(inputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'rb')
	else:
		f = open(inputFileName, 'rb')
	size = struct.unpack('ii', f.read(8))	
	data = numpy.fromfile(file = f, dtype=numpy.float32).reshape(size, order='C')
	if(transpose == 'N'):
		return data
	elif(transpose == 'T'):
		return data.transpose()
	else:
		print "read T or N?"


def readIntTensor(inputFileName):
	import numpy
	import gzip
	import struct
	if(inputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'rb')
	else:
		f = open(inputFileName, 'rb')
	size = struct.unpack('ii', f.read(8))
	data = numpy.fromfile(file = f, dtype=numpy.int32).reshape(size, order='F')
	return data

def readCIntTensor(inputFileName):
	import numpy
	import gzip
	import struct
	if(inputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'rb')
	else:
		f = open(inputFileName, 'rb')
	size = struct.unpack('ii', f.read(8))
	data = numpy.fromfile(file = f, dtype=numpy.int32).reshape(size, order='C')
	return data


def readC1dIntTensor(inputFileName):
	import numpy
	import gzip
	import struct
	if(inputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'rb')
	else:
		f = open(inputFileName, 'rb')
	size = struct.unpack('i', f.read(4))
	data = numpy.fromfile(file = f, dtype=numpy.int32)
	data = data.reshape((data.shape[0] / size[0], size[0]), order='C')
	return data


def writeTensor(data, outputFileName):
	import numpy
	import struct
	import gzip
	if(outputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'wb')
	else:
		f = open(outputFileName, 'wb')
	f.write(struct.pack('ii', data.shape[0], data.shape[1]))
	for j in range(data.shape[1]):
		data[:, j].tofile(f, sep="", format="%f")
	f.close()

# writeCTensor, C format Row major order => default format for ngram
def writeCTensor(data, outputFileName):
	import numpy
	import struct
	import gzip
	if(outputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'wb')
	else:
		f = open(outputFileName, 'wb')
	f.write(struct.pack('ii', data.shape[0], data.shape[1]))
	for i in range(data.shape[0]):
		data[i, :].tofile(f, sep="", format="%f")
	f.close()

def writeCIntTensor(data, outputFileName):
	import numpy
	import struct
	import gzip
	if(outputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'wb')
	else:
		f = open(outputFileName, 'wb')
	f.write(struct.pack('ii', data.shape[0], data.shape[1]))
	for i in range(data.shape[0]):
		data[i, :].tofile(f, sep="", format="%i")
	f.close()

def writeC1dIntTensor(data, outputFileName):
	import numpy
	import struct
	import gzip
	if(outputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'wb')
	else:
		f = open(outputFileName, 'wb')
	f.write(struct.pack('i', data.shape[1]))
	for i in range(data.shape[0]):
		data[i, :].tofile(f, sep="", format="%i")
	f.close()



def writeIntTensor(data, outputFileName):
	import numpy
	data = numpy.array(data, dtype=numpy.int32)
	import numpy
	import struct
	import gzip
	if(outputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'wb')
	else:
		f = open(outputFileName, 'wb')	
	f.write(struct.pack('ii', data.shape[0], data.shape[1]))
	for j in range(data.shape[1]):
		data[:, j].tofile(f, sep="", format="%i")
	f.close()

def readSize(inputFileName):
	import numpy
	import gzip
	import struct
	if(inputFileName[-3:] == '.gz'):
		f = gzip.open(inputFileName, 'rb')
	else:
		f = open(inputFileName, 'rb')
	sizeT = struct.unpack('ii', f.read(8))
	size = [sizeT[0], sizeT[1]]
	return size
def plainTensor(inputFileName, outputFileName):
	readTensor(inputFileName)
	writePlainTensor(outputFileName)
def plainIntTensor(inputFileName, outputFileName):
	readIntTensor(inputFileName)
	writePlainIntTensor(outputFileName)

