#!/usr/bin/env python
import sys
if (len(sys.argv) != 6):
	print('codeWordFileName outputNetworkSizeFileName vocFileName mapShortlistFileName prefixOutput')
else:
	import os
	import numpy
	from soul.libdata import *
	codeWordFileName = sys.argv[1]
	outputNetworkSizeFileName = sys.argv[2]
	voc = open(sys.argv[3], 'r').readlines()
	mapShortlist = map(int, open(sys.argv[4], 'r').readlines())
	prefixOutput = sys.argv[5]
	outCodeWordFileName = prefixOutput + 'codeWord'
	outOutputNetworkSizeFileName = prefixOutput + 'outputNetworkSize'
	i = 0
	code = readIntTensor(codeWordFileName)
	size = readIntTensor(outputNetworkSizeFileName)
	for idW in range(len(mapShortlist)):
		if(mapShortlist[idW] == -1):
			print(idW)
		else:
			code[mapShortlist[idW], :] = -1
			i = i + 1

	x = voc.index('<s>\n')
	code[x, :] = -1
	x = voc.index('</s>\n')
	code[x, :] = -1
	x = voc.index('<UNK>\n')
	code[x, :] = -1
	# -3 because in shortlist, we don't have <s>, </s>, <UNK>, if having => bug
	newSize = int(size[0]) - i - 3
	size[0] = newSize
	aWId = []	
	for i in range(len(voc)): #With alone word not in shortlist
		if(code[i, 2] == -1 and code[i, 1] != -1):
			aWId.append(i)
	nWId = len(aWId)
	for iWId in range(nWId):
		code[aWId[iWId], 1] = newSize - nWId + iWId

	writeIntTensor(size, outOutputNetworkSizeFileName)
	writeIntTensor(code, outCodeWordFileName)
