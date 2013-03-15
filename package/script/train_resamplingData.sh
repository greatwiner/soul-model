#!/bin/bash
if [ "$#" != "3" ]
then
	echo 'configFile order type(all, sl, out)'
	exit
fi
. $1
order=$2
type=$3

if [ "$type" == "all" ]
then
	dirEx=$order'gram/all'
elif [ "$type" == "sl" ]
then
	dirEx=$order'gram/'$shortlistSize'sl'
elif [ "$type" == "out" ]
then
	dirEx=$order'gram/out'$shortlistSize'sl'
fi

mkdir -p $dirEx
echo "Do ex "$dirEx
echo "resampling"
hostname >> $dirEx/resampling.log
date >> $dirEx/resampling.log

if [ "$type" == "all" ]
then
	echo "resamplingData.exe $dataTrainDes $voc $voc $order 1 1 $dirEx/train_ 1 $lastEpoch"
	echo "resamplingData.exe $dataTrainDes $voc $voc $order 1 1 $dirEx/train_ 1 $lastEpoch"  >> $dirEx/resampling.log
	resamplingData.exe $dataTrainDes $voc $voc $order 1 1 $dirEx/train_ 1 $lastEpoch  >> $dirEx/resampling.log
elif [ "$type" == "sl" ]
then
	echo "resamplingData.exe $shortlistDataTrainDes $voc $shortlist $order 1 0 $dirEx/train_ 1 $shortlistLastEpoch"
	echo "resamplingData.exe $shortlistDataTrainDes $voc $shortlist $order 1 0 $dirEx/train_ 1 $shortlistLastEpoch"  >> $dirEx/resampling.log
	resamplingData.exe $shortlistDataTrainDes $voc $shortlist $order 1 0 $dirEx/train_ 1 $shortlistLastEpoch  >> $dirEx/resampling.log
elif [ "$type" == "out" ]
then
	echo "resamplingData.exe $outDataTrainDes $voc $outVoc $order 1 0 $dirEx/train_ 1 $outLastEpoch"
	echo "resamplingData.exe $outDataTrainDes $voc $outVoc $order 1 0 $dirEx/train_ 1 $outLastEpoch"  >> $dirEx/resampling.log
	resamplingData.exe $outDataTrainDes $voc $outVoc $order 1 0 $dirEx/train_ 1 $outLastEpoch  >> $dirEx/resampling.log
fi

date >> $dirEx/resampling.log
