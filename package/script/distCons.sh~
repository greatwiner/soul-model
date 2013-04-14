#!/bin/bash

if [ $# -ne 4 ]; then
    echo "directory start end destination"
    exit
fi

directory=$1
start=$2
end=$3
destination=$4
key1="0OutputWeight0"
key2="OutputWeight0"

for (( i = $start; i<=$end-1; i++))
do
    echo "absoluteDist.exe $directory$i$key2 $directory$(($i+1))$key2 >> $destination"
    absoluteDist.exe $directory$i$key2 $directory$(($i+1))$key2 >> $destination
done