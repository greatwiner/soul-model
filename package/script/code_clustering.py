#!/usr/bin/env python
THRESHOLDGROUP=1000
def baseN(num, base, level):
	baseR = []
	n = num
	itera = 0
	while(itera!= level):
		m = n / base
		c = n % base
		baseR.append(c)	
		n = m
		itera = itera + 1
	baseR.reverse()
	return baseR
def createLayer(wordNumber, base): #Create randomly 
	import math
	import numpy
	level = int(math.ceil(math.log(wordNumber, base)))
	layerNumber = (base ** level - 1) / (base - 1)
	offset = numpy.zeros(level)
	for i in range(1, level):
		offset[i] = offset[i - 1] * base + 1
	#size = numpy.ones(layerNumber, dtype=int) * base
	size = [base] * layerNumber

	baseLayerNumber = base ** (level - 1)
	x = wordNumber // baseLayerNumber
	y = wordNumber % baseLayerNumber

	tempSize = numpy.ones(baseLayerNumber, dtype=int) * x
	randomAdd = numpy.zeros(baseLayerNumber, dtype=int)
	randomAdd[:y] = 1
	idRandomAdd = numpy.random.permutation(baseLayerNumber)
	for i in range(baseLayerNumber):
		tempSize[i] = tempSize[i] + randomAdd[idRandomAdd[i]]
	size[-baseLayerNumber:] = tempSize[:]
	code = numpy.zeros((wordNumber, 2 * level))
	k = 0
	correct1 = 0
	codeXX = numpy.zeros(2 * level, dtype=int)
	for i in range(baseLayerNumber):
		codeX = baseN(i * base, base, level)
		for j in range(0, level - 1):
			codeXX[j * 2 + 1] = codeX[j]
		codeXX[0] = 0
		for j in range(1, level - 1):
			codeXX[j * 2] = offset[j] +  (codeXX[j * 2 - 2] - offset[j - 1]) * base + codeXX[j * 2 - 1]
		if(tempSize[i] == 1):
			code[k,:] = codeXX[:]
			code[k, -2:] = -1
			k = k + 1
			correct1 = correct1 + 1
		else:
			j = level - 1
			codeXX[j * 2] = offset[j] +  (codeXX[j * 2 - 2] - offset[j - 1]) * base + codeXX[j * 2 - 1] - correct1
			for j in range(int(tempSize[i])):
				code[k,:] = codeXX[:]
				code[k, -1] = j
				k = k + 1
	#Delete class with size = 1
	finish = 0
	while(not finish):
		try:
			size.remove(1)
		except ValueError:		
			finish = 1
	ipermutation = numpy.random.permutation(wordNumber)
	codeOut = numpy.zeros((wordNumber, 2 * level), dtype=int)
	for i in range(wordNumber):
		codeOut[i, :] = code[ipermutation[i], :]
	return numpy.array(size), codeOut


import sys

if (len(sys.argv) != 7):
	print('inputFeatureFileName voc shortlist groupNumber iterationNumber prefixOutput')
else:

	import os
	import numpy
	import scipy.cluster.vq
	from soul.libdata import *
	inputFeatureFileName = sys.argv[1]
	voc = open(sys.argv[2], 'r').readlines()
	shortlistFileName = sys.argv[3]
	if(os.path.exists(shortlistFileName)):
		shortlist = open(shortlistFileName, 'r').readlines()
		skipNumber = len(shortlist)
		shortlistIndex = []
		for iS in range(skipNumber):
			try:
				x = voc.index(shortlist[iS])
				shortlistIndex.append(x)
			except ValueError:
				skipNumber = skipNumber - 1				
				print('WARNING: ' + shortlist[iS] + ' is not in voc')
	else:
		shortlist = []
		skipNumber = 0
		shortlistIndex = []
	groupNumber = int(sys.argv[4])
	iterationNumber = int(sys.argv[5])
	prefixOutput = sys.argv[6]
	codeWordName = prefixOutput + 'codeWord'
	outputNetworkSizeName = prefixOutput + 'outputNetworkSize'
	codFileName = prefixOutput + 'cod'
	wordNumber = len(voc)
	dimension = 4
	#Read feature
	size = readSize(inputFeatureFileName)
	if (size[0] == wordNumber):
		feature = readTensor(inputFeatureFileName, 'N')
	else:
		feature = readTensor(inputFeatureFileName, 'T')
	dimension = feature.shape[1]
	print('Kmeans')
	center, label  = scipy.cluster.vq.kmeans2(feature, groupNumber , iter=iterationNumber, thresh=1.0000000000000001e-05, minit='points', missing='warn')
	print('Modify group')
	SSINDEX = voc.index("<s>\n")
	ESINDEX = voc.index("</s>\n")
	UNKINDEX = voc.index("<UNK>\n")
	groupNumber = groupNumber + 1
	# Make words in shortlist and SS, ES, UNK out of group, -> last group
	for iS in range(skipNumber):
		label[shortlistIndex[iS]] = groupNumber - 1
	label[SSINDEX] = groupNumber - 1
	label[ESINDEX] = groupNumber - 1
	label[UNKINDEX] = groupNumber - 1
	numGroup = [0] * (groupNumber)
	mapGroup = [-1] * (groupNumber)
	for j in range(wordNumber):
		numGroup[label[j]] = numGroup[label[j]] + 1	
	newGroupNumber = 0
	#Delete group with 0 member, 1 member. In 1 moves to last group	
	idOneGroup = []
	for i in range(groupNumber):
		if(numGroup[i] == 1): #Like skipped words, mark group
			#print("one alone")
			idOneGroup.append(i)			
			numGroup[i] = 0
		elif(numGroup[i] != 0):
			mapGroup[i] = newGroupNumber
			newGroupNumber = newGroupNumber + 1
	
	for i in range(wordNumber):
		if(label[i] in idOneGroup): #Move alone into last group 
			label[i] = mapGroup[groupNumber - 1]
			numGroup[groupNumber - 1] = numGroup[groupNumber - 1] + 1
		else:
			label[i] = mapGroup[label[i]]
	# Remap numGroup (member number of group)
	newNumGroup = [1] * newGroupNumber
	for i in range(groupNumber):
		if(numGroup[i] > 1):		
			newNumGroup[mapGroup[i]] = numGroup[i]
	#We have now real localCenter and label and groupNumber
	numGroup = newNumGroup
	groupNumber = newGroupNumber
	lever1groupNumber = groupNumber - 1 + numGroup[groupNumber - 1] # Main softmax = group number + skipped words number
	lever1offset = groupNumber - 1
	labelLonely = groupNumber - 1
	grand = {}
	localSize = {}
	localCode = {}
	localId = {}
	idGrand = 0
	offset = {}
	run = groupNumber - 1
	groupNumber = groupNumber - 1
	del numGroup[-1]
	for i in range(run): #Only from 0 to groupNumber - 1 because groupNumber has skipped members
		if(numGroup[i] > THRESHOLDGROUP):
			# use createLayer with base = int(numGroup[i] ** 0.5) + 1 => add only one more level to have 3 levels
			localSize[idGrand], localCode[idGrand] = createLayer(numGroup[i], int(numGroup[i] ** 0.5) + 1)
			dimension = localCode[idGrand].shape[1] + 2
			numGroup[i] = localSize[idGrand][0]
			grand[idGrand] = i
			numGroup.append(1) #Hack to copy only
			numGroup[-1:]= localSize[idGrand][1:]
			offset[idGrand] = groupNumber			
			groupNumber = groupNumber + int(localSize[idGrand][0])
			localId[idGrand] = 0
			idGrand = idGrand + 1
	code = numpy.ones((wordNumber, dimension)) * -1
	numGrand = idGrand
	idL = numpy.zeros(groupNumber + 1)
	cLonely = 0
	for i in range(wordNumber):
		code[i, 0] = 0
		if(label[i] == labelLonely): # So lonely
			code[i, 1] = lever1offset + cLonely
			cLonely = cLonely + 1
		else:
			code[i, 1] = label[i]
			blGrand = 0
			for idGrand in range(numGrand):
				if(label[i] == grand[idGrand]):
					code[i, 2] = code[i, 1] + 1
					code[i, 3] = localCode[idGrand][localId[idGrand], 1]
					code[i, 4] = localCode[idGrand][localId[idGrand], 2] + offset[idGrand]
					code[i, 5] = localCode[idGrand][localId[idGrand], 3]
					localId[idGrand] = localId[idGrand] + 1
					blGrand = 1
					break
			if(not blGrand):
				code[i, 2] = code[i, 1] + 1
				code[i, 3] = idL[code[i, 2]]
				idL[code[i, 2]] = idL[code[i, 2]] + 1
	print(numGrand, grand)
	#Output

	numGroup[:0] = [lever1groupNumber]
	numGroup = numpy.array(numGroup, dtype=numpy.int32)
	numGroup.resize((numGroup.shape[0], 1))
	writeIntTensor(code, codeWordName)	
	writeIntTensor(numGroup, outputNetworkSizeName)
