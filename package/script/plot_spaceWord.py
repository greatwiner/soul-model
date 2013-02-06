#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font
import numpy
from soul.libdata import *
import codecs

if (len(sys.argv) !=  5):
	print('spaceFileName vocFileName listFileName outputFileName')
else:
	spaceFileName = sys.argv[1]
	vocFileName = sys.argv[2]
	listWordsFileName = sys.argv[3]
	outputFileName = sys.argv[4]
	print("Read")
	feature = readTensor(spaceFileName, 'N')

	voc = codecs.open(vocFileName, 'r', 'utf-8').read().split()
	listWords = codecs.open(listWordsFileName, 'r', 'utf-8').read().split()
	indexWords = []
	for iLW in range(len(listWords)):
		indexWords.append(voc.index(listWords[iLW]))
	print("Draw")
	plt.hold(True)
	style = ['.b', ',g', 'or', 'vc', '^m', '<y', '>k', '1b', '2g', '3r', '4c', 'sm', 'py', '*k', 
                 '.g', ',r', 'oc', 'vm', '^y', '<k', '>b', '1g', '2r', '3c', '4m', 'sy', 'pk', '*b']
	for iPlot in range(len(listWords)):
		if(iPlot % 500 == 0):
			sys.stdout.write(str(iPlot) + " ... ")
			sys.stdout.flush()
		plt.plot(feature[indexWords[iPlot], 0], feature[indexWords[iPlot], 1], style[0], markersize = 1)
	print("\nAnnotate")
	pt = font.FontProperties(size=2)
	for iPlot in range(len(listWords)):
		if(iPlot % 10000 == 0):
			sys.stdout.write(str(iPlot) + " ... ")
			sys.stdout.flush()
		plt.annotate( voc[indexWords[iPlot]], [feature[indexWords[iPlot],0], feature[indexWords[iPlot],1]], xycoords='data', color = "r", fontproperties = pt)
	print
	plt.savefig(outputFileName, dpi = 480)
