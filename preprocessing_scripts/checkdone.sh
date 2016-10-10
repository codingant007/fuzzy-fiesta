#!/bin/bash
total_in=0
total_out=0
total_keys=0
for dir in `ls -d */`
do
	i=0
	for file in `ls $dir[a-zA-Z]*.JPEG`
	do
		i=$(($i+1))
	done
	total_in=$(($total_in+$i))
	
	j=0
	for file in `ls $dir[0-9]*.JPEG`
	do
		j=$(($j+1))
	done
	total_out=$((total_out+$j))

	k=0
	for file in `ls $dir[0-9]*.key`
	do
		k=$(($k+1))
	done
	total_keys=$((total_keys+$k))

	echo $dir"input -> "$i
	echo $dir"ouput -> "$j
	echo $dir"keys -> "$k
	if [ $i -eq $j -a $i -eq $k ]; then
		echo $dir" complete"
	else
		echo $dir" not complete"
	fi
	echo "-----------------------------"
done

echo  "total in -> "$total_in
echo "total out -> "$total_out
echo "total keys -> "$total_keys
if [ $total_in -eq $total_out -a $total_in -eq $total_keys ] ; then
	echo "Total complete"
else
	echo "Total not complete"
fi

