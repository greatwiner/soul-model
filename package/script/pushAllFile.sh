#!/bin/bash

if [ $# -ne 4 ]; then
    echo "directory start end what"
    exit
fi

directory=$1
start=$2
end=$3

what=$4

for (( i=$start; i<=$end; i++ ))
do
    echo "pushAllParameter.exe $directory$i $directory$i $what"
    pushAllParameter.exe $directory$i $directory$i $what
done