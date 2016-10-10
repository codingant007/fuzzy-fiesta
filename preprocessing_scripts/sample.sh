#!/bin/bash
i=0
function process_image {
	local file=$1
	local class_dir=$2
	local i=$3
		echo $file" to -> "$i".JPEG"
		convert -resize 256x256\! $file $class_dir$i".JPEG"
		convert $class_dir$i".JPEG" $class_dir$i".pgm"
		../siftDemoV4/sift <../dataset/$class_dir$i".pgm" >../dataset/$class_dir$i".key"
		if [ $? -ne 0 ]; then
			echo "interest point extraction failed for "$class_dir$i >> $class_dir"error.log"
		fi
		#python sample.py $class_dir$i".JPEG" $class_dir$i".key"
		if [ $? -ne 0 ]; then
			echo "image processing failed for "$class_dir$i >> $class_dir"error.log"
		fi
}

function process_image_new {
	local file=$1
	local class_dir=$2
	local i=$3
	echo $file" to -> "$i".JPEG"
	convert -resize 256x256\! $file $class_dir$i".JPEG"
	./sift_ip_sample $class_dir$i".JPEG" $class_dir$i".key"
	if [ $? -ne 0 ]; then
		echo "sift failed for "$class_dir$i >> $class_dir"error.log"
	fi
	python sample_img.py $class_dir$i".JPEG" $class_dir$i".key"
	if [ $? -ne 0 ]; then
		echo "image processing failed for "$class_dir$i >> $class_dir"error.log"
	fi
}

START=$(date +%s);

for class_dir in `ls -d */`;
do
	echo $class_dir
	rm $class_dir"process.log" $class_dir"error.log"

	for file in `ls $class_dir[a-zA-Z]*.JPEG`;
	do
		i=$(($i+1))
		process_image_new $file $class_dir $i&
		if (( $i % 100 == 0 )); then
			wait
			END=$(date +%s);
			echo "--------------------------------------------------------"
			echo -n $i" at "
			echo $((END-START)) | awk '{print int($1/60)"min:"int($1%60)"sec"}'
			echo "--------------------------------------------------------" >> $class_dir"process.log"
			echo -n $i" at " >> $class_dir"process.log"
			echo $((END-START)) | awk '{print int($1/60)"min:"int($1%60)"sec"}' >> $class_dir"process.log"
		fi
	done
done
wait
END=$(date +%s);
echo $((END-START)) | awk '{print "Program completed at "int($1/60)"min:"int($1%60)"sec"}'