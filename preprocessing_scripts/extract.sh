#!/bin/bash

for file in `ls *.tar`;
do
	echo $file
	class=`echo $file | sed 's/^.*_\(.*\)\.tar$/\1/g'`
	mkdir $class
	tar -xvf $file -C $class&
done