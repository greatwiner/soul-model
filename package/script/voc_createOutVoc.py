#!/usr/bin/env python
import sys
if (len(sys.argv) != 4):
	print('vocFileName shortlistFileName outputVocFileName')
else:
	import os
	import numpy
	voc = open(sys.argv[1], 'r').readlines()
	shortlist = open(sys.argv[2], 'r').readlines()
	i = 0
	for idW in range(len(shortlist)):
		try:
			x = voc.index(shortlist[idW])
			voc[x] = 'prefix.' + shortlist[idW]
			i = i + 1
		except ValueError:
			print shortlist[idW]
	try:
		x = voc.index('<s>\n')
		voc[x] = 'prefix.<s>\n'
	except ValueError:
		print('error: <s>')
	try:
		x = voc.index('</s>\n')
		voc[x] = 'prefix.</s>\n'
	except ValueError:
		print('error: </s>')
	try:
		x = voc.index('<UNK>\n')	
		voc[x] = 'prefix.<UNK>\n'
	except ValueError:
		print('error: <UNK>')
	outputFile = open(sys.argv[3], 'w')
	outputFile.write(''.join(voc))
	outputFile.close()

