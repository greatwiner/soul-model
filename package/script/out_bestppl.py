#!/usr/bin/env python
import os
import sys
import glob
import numpy
if (len(sys.argv) != 2):
	print("inputDir")
else:
	fileList = []
	rootdir = sys.argv[1]
	filelog = 'train.log'
	keyWord = 'With' 
	order = -3
	for root, subFolders, files in os.walk(rootdir):
	    	for mfile in files:
			if(mfile == filelog):
	        		fileList.append(os.path.join(root,mfile))
	for mfile in fileList:
		prc = os.popen('cat ' + mfile + ' | grep \'' + keyWord + '\' | uniq').readlines()
		mfile = mfile + ' ' * (50 - len(mfile)) + '\t'
		sys.stdout.write(mfile)
		data = {}
		for line in prc:
			splitLine = line.split()			
			x = float(splitLine[order])
			data[int(splitLine[2][:-1])] = x
		m = 10000000
		im = -1
		for i in range(len(data)):
			if(m > data[i]):
				m = data[i]
				im = i
		sys.stdout.write("%d %.2f" % (im, m))
		print

