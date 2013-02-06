#!/usr/bin/env python
import sys
if (len(sys.argv) != 5):
	print('featureFileName transpose dim outputFileName')
	print('transpose: N or T')
else:
	import os
	import numpy
	from soul.libdata import *
	featureFileName = sys.argv[1]
	transpose = sys.argv[2]
	dim = int(sys.argv[3])
	outputFileName = sys.argv[4]
	feature = readTensor(featureFileName, transpose) #Transpose
	feature = numpy.array(feature, dtype=numpy.float64)
	import mdp
	pcanode = mdp.nodes.PCANode(output_dim = dim)
	pcanode.train(feature)
	pcanode.stop_training()
	pcaFeature = pcanode.execute(feature)
	pcaFeature = numpy.array(pcaFeature, dtype=numpy.float32)
	writeTensor(pcaFeature, outputFileName)
