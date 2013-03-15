#!/bin/bash
if [ "$#" != "3" ]
then
	echo 'configFile order type(all, sl, out)'
	exit
fi
. $1
order=$2
rtype=$3

if [ "$rtype" == "all" ]
then
	outDirEx=$order'gram/out'$shortlistSize'sl'
	slDirEx=$order'gram/'$shortlistSize'sl'
	dirEx=$order'gram/all'$shortlistSize'sl'
	foDirEx=$firstOrder'gram/all'$shortlistSize'sl'
elif [ "$rtype" == "sl" ]
then
	dirEx=$order'gram/'$shortlistSize'sl'
elif [ "$rtype" == "out" ]
then
	slDirEx=$order'gram/'$shortlistSize'sl'
	dirEx=$order'gram/out'$shortlistSize'sl'
fi

echo "Do ex "$dirEx
echo "test prototype"
echo "text2Perplexity.exe $dirEx/0 $blockSize $validation $validType"
text2Perplexity.exe $dirEx/0 $blockSize $validation $validType
