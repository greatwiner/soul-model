#!/bin/bash

if [ $# -ne 3 ]; then
    echo "indRetain directory1 directory2"
    exit
fi

file=0
echo "mkdir -p $3"
mkdir -p $3
echo "cp $20 $3$file"
cp $20 $3$file
while read line
do
    file=$(($file+1))
    echo "cp $2$line $3$file"
    cp $2$line $3$file
done < $1

echo $file