#!/bin/bash

for dir in `ls -d */`
do
	echo $dir
	rm $dir[0-9]*.*&
done