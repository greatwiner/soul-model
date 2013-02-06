#!/bin/bash
if [ "$#" != "3" ]
then
	echo 'configFile order type(all, sl, out)'
	exit
fi
. $1
order=$2
type=$3

minEpoch=1
if [ "$type" == "all" ]
then
	dirEx=$order'gram/all'$shortlistSize'sl'	
	dirData=$mainData/$order'gram/all'
	maxEpoch=$lastEpoch
elif [ "$type" == "sl" ]
then
	dirEx=$order'gram/'$shortlistSize'sl'
	dirData=$mainData/$dirEx
	maxEpoch=$shortlistLastEpoch
elif [ "$type" == "out" ]
then
	dirEx=$order'gram/out'$shortlistSize'sl'
	dirData=$mainData/$dirEx
	maxEpoch=$outLastEpoch
fi

mkdir -p $dirEx
echo "Do ex "$dirEx
echo "train"
hostname >> $dirEx/train.log
date >> $dirEx/train.log

echo sequenceTrain.exe $dirEx/ $dirData/train_ $maxExampleNumber $validation $validType $lrt $minEpoch $maxEpoch 
echo sequenceTrain.exe $dirEx/ $dirData/train_ $maxExampleNumber $validation $validType $lrt $minEpoch $maxEpoch >> $dirEx/train.log
sequenceTrain.exe $dirEx/ $dirData/train_ $maxExampleNumber $validation $validType $lrt $minEpoch $maxEpoch >> $dirEx/train.log

date >> $dirEx/train.log


