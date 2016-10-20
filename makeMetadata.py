# Standalone script for generating pickle file of metadata
# Generates alldata.pkl, train.pkl, val.pkl, test.pkl 
# Generates alldata_group.pkl, train_group.pkl, val_group.pkl, test_group.pkl
# usage - makeMetadata.py <dataset path>

import six.moves.cPickle as pickle
import os,sys
from os.path import isfile, join
import re
import random

dataset_path = sys.argv[1]

def insert_groupinfo(sample,class_name,label_number,group_number,group_info):
	grp_sample_list = []
	if group_number in group_info.keys():
		assert group_info[group_number][0] == group_number
		assert group_info[group_number][1] == class_name
		assert group_info[group_number][2] == label_number
		grp_sample_list = group_info[group_number][3]
	grp_sample_list += [sample]
	group_info[group_number] = (group_number,class_name,label_number,grp_sample_list)
	return group_info

def getSamplesInfo():
	sample_list = []
	group_info = {}
	label_number = 0
	for class_name in os.listdir(dataset_path):
		if not isfile(join(dataset_path,class_name)):
			print class_name
			label_number = label_number+1
			for sample in os.listdir(join(dataset_path,class_name)):
				if not sample.find('-') < 0:
					sample_path = join(join(dataset_path,class_name),sample)
					group_number = sample.split('-')[0]

					group_info = insert_groupinfo(sample_path ,class_name ,label_number ,group_number ,group_info)
					sample_list.append((sample_path,class_name,label_number,group_number))
	return sample_list,group_info

def partition(sample_list, proportion=(0.7,0.1,0.2)):
	total_size = len(sample_list)
	traning_size = int(total_size*proportion[0])
	validation_size = int(total_size*proportion[1])
	training_set = sample_list[:traning_size]
	validation_set = sample_list[traning_size:traning_size+validation_size]
	test_set = sample_list[traning_size+validation_size:]
	return training_set,validation_set,test_set

def get_samples_from_groups(group_list):
	sample_list = []
	for group in group_list:
		for sample in group[3]:
			sample_element = (sample,group[1],group[2],group[0])
			sample_list += [sample_element]
	return sample_list


def saveSamplesInfo(alldata_file="metadata/alldata.pkl",train_file="metadata/train.pkl",val_file="metadata/val.pkl",test_file="metadata/test.pkl", \
					alldata_group_file="metadata/alldata_group.pkl",train_group_file="metadata/train_group.pkl",val_group_file="metadata/val_group.pkl",test_group_file="test_group.pkl"):

	sample_list,group_info = getSamplesInfo()
	group_list = group_info.values()

	print "No of Groups: ",len(group_list)
	
	random.shuffle(sample_list)
	random.shuffle(group_list)

	train_set,val_set,test_set = partition(group_list)

	print "No of Train groups: ",len(train_set)
	print "No of Val groups: ",len(val_set)
	print "No of Test groups: ",len(test_set)

	train_samples = get_samples_from_groups(train_set)
	val_samples = get_samples_from_groups(val_set)
	test_samples = get_samples_from_groups(test_set)

	print "No of Train Samples: ",len(train_samples)
	print "No of Val Samples: ",len(val_samples)
	print "No of Test groups: ",len(test_samples)
	print "Sample list length: ", len(sample_list)
	print "Total Number of samples: ",(len(train_samples)+len(val_samples)+len(test_samples))

	assert len(sample_list) == len(train_samples)+len(val_samples)+len(test_samples)

	with open(train_file,'wb') as f:
		pickle.dump(train_samples , f, pickle.HIGHEST_PROTOCOL)
	with open(val_file,'wb') as f:
		pickle.dump(val_samples , f, pickle.HIGHEST_PROTOCOL)
	with open(test_file,'wb') as f:
		pickle.dump(test_samples , f, pickle.HIGHEST_PROTOCOL)
	with open(alldata_file,'wb') as f:
		pickle.dump(sample_list , f, pickle.HIGHEST_PROTOCOL)

	with open(train_group_file,'wb') as f:
		pickle.dump(train_set , f, pickle.HIGHEST_PROTOCOL)
	with open(val_group_file,'wb') as f:
		pickle.dump(val_set , f, pickle.HIGHEST_PROTOCOL)
	with open(test_group_file,'wb') as f:
		pickle.dump(test_set , f, pickle.HIGHEST_PROTOCOL)
	with open(alldata_group_file,'wb') as f:
		pickle.dump(group_list , f, pickle.HIGHEST_PROTOCOL)

	print "Metadata saved to eight pickle files in metadata/",


if __name__ == '__main__':
	saveSamplesInfo()