#!/usr/bin/env python
import os
import sys
import glob
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
			#if(x > 10000):
			#	x = 9999
			data[int(splitLine[2][:-1])] = x
		for key in data:
			sys.stdout.write("%.2f " % data[key])
		print

