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
elif [ "$rtype" == "sl" ]
then
	dirEx=$order'gram/'$shortlistSize'sl'
elif [ "$rtype" == "out" ]
then
	slDirEx=$order'gram/'$shortlistSize'sl'
	dirEx=$order'gram/out'$shortlistSize'sl'
fi

mkdir -p $dirEx
echo "Do ex "$dirEx
echo "create"
hostname >> $dirEx/create.log
date >> $dirEx/create.log

if [ "$rtype" == "sl" ]
then
	echo createPrototype.exe $type $voc $shortlist 1 0 $order $m $nonLinearType $H xxx xxx $dirEx/0
	echo createPrototype.exe $type $voc $shortlist 1 0 $order $m $nonLinearType $H xxx xxx $dirEx/0 >> $dirEx/create.log
	createPrototype.exe $type $voc $shortlist 1 0 $order $m $nonLinearType $H xxx xxx $dirEx/0 >> $dirEx/create.log
	echo -e "5 1\n$lr\n$lrdc\n$wd\n$blockSize\n$divide"> $dirEx/0.par
elif [ "$rtype" == "out" ]
then
	if [ ! -e $slDirEx/$shortlistLastEpoch ]
	then
		echo ERROR $slDirEx/$shortlistLastEpoch
		exit
	fi		
	inModelFileName=$slDirEx/$shortlistLastEpoch
	codeWord=$inModelFileName'_LookupTable'.pca$nPCA.$group.codeWord
	outputNetworkSize=$inModelFileName'_LookupTable'.pca$nPCA.$group.outputNetworkSize
	if [ -e $inModelFileName'_'LookupTable.pca$nPCA.$group.outputNetworkSize ]
	then
		echo "Don't do Kmeans"
	else
		if [ ! -e $inModelFileName'_'LookupTable.pca$nPCA ]
		then
			if [ ! -e $inModelFileName'_'LookupTable ]
			then
				echo "pushAllParameter.exe $inModelFileName $inModelFileName""_ l"
				echo "pushAllParameter.exe $inModelFileName $inModelFileName""_ l" >> $dirEx/create.log
				pushAllParameter.exe $inModelFileName $inModelFileName'_' l  >> $dirEx/create.log
			fi
			echo code_pcaFeature.py $inModelFileName'_'LookupTable T $nPCA $inModelFileName'_'LookupTable.pca$nPCA
			echo code_pcaFeature.py $inModelFileName'_'LookupTable T $nPCA $inModelFileName'_'LookupTable.pca$nPCA >> $dirEx/create.log
			code_pcaFeature.py $inModelFileName'_'LookupTable T $nPCA $inModelFileName'_'LookupTable.pca$nPCA  >> $dirEx/create.log
		fi
		if [ $skip == 0 ]
		then
			shortlist='xxx'
		fi
		echo code_clustering.py $inModelFileName'_'LookupTable.pca$nPCA $voc $shortlist $group $nKmeans $inModelFileName'_'LookupTable.pca$nPCA.$group.
		echo code_clustering.py $inModelFileName'_'LookupTable.pca$nPCA $voc $shortlist $group $nKmeans $inModelFileName'_'LookupTable.pca$nPCA.$group. >> $dirEx/create.log
		code_clustering.py $inModelFileName'_'LookupTable.pca$nPCA $voc $shortlist $group $nKmeans $inModelFileName'_'LookupTable.pca$nPCA.$group.
		date >> $dirEx/create.log
	fi
	saveModelFileName=$dirEx/0
	echo "create out paras file"
	outCodeWord=$inModelFileName'_LookupTable'.pca$nPCA.$group.out.codeWord
	outOutputNetworkSize=$inModelFileName'_LookupTable'.pca$nPCA.$group.out.outputNetworkSize
	if [ ! -e $outCodeWord ]
	then
		echo code_createOutCode.py $codeWord $outputNetworkSize $voc $mapshortlist $inModelFileName'_LookupTable'.pca$nPCA.$group.out.
		echo code_createOutCode.py $codeWord $outputNetworkSize $voc $mapshortlist $inModelFileName'_LookupTable'.pca$nPCA.$group.out. >> $dirEx/create.log
		code_createOutCode.py $codeWord $outputNetworkSize $voc $mapshortlist $inModelFileName'_LookupTable'.pca$nPCA.$group.out. >> $dirEx/create.log
	fi
	echo "growPredictionSpace"
	echo growPredictionSpace.exe $inModelFileName $outVoc 1 0 $outCodeWord $outOutputNetworkSize $saveModelFileName
	echo growPredictionSpace.exe $inModelFileName $outVoc 1 0 $outCodeWord $outOutputNetworkSize $saveModelFileName >> $dirEx/create.log
	growPredictionSpace.exe $inModelFileName $outVoc 1 0 $outCodeWord $outOutputNetworkSize $saveModelFileName 
	echo -e "5 1\n$lr\n$lrdc\n$wd\n$blockSize\n$divide"> $dirEx/0.par
elif [ "$rtype" == "all" ]
then
	echo -e "5 1\n$lr\n$lrdc\n$wd\n$blockSize\n$divide"> $dirEx/0.par
	if [ ! -e $slDirEx/$shortlistLastEpoch ]
	then
		echo ERROR $slDirEx/$shortlistLastEpoch
		exit
	fi		
	if [ ! -e $outDirEx/$outLastEpoch ]
	then
		echo ERROR $outDirEx/$outLastEpoch
		exit
	fi		
	inModelFileName=$slDirEx/$shortlistLastEpoch
	codeWord=$inModelFileName'_LookupTable'.pca$nPCA.$group.codeWord
	outputNetworkSize=$inModelFileName'_LookupTable'.pca$nPCA.$group.outputNetworkSize
	echo "growOutPredictionSpace"
	echo growOutPredictionSpace.exe  $inModelFileName  $outDirEx/$outLastEpoch $dirEx/0
	echo growOutPredictionSpace.exe  $inModelFileName  $outDirEx/$outLastEpoch $dirEx/0 >> $dirEx/create.log
	growOutPredictionSpace.exe  $inModelFileName  $outDirEx/$outLastEpoch $dirEx/0 >> $dirEx/create.log
fi
