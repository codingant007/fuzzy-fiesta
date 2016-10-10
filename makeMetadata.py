# Standalone script for generating pickle file of metadata
# usage - makeMetadata.py <dataset path>

import six.moves.cPickle as pickle
import os,sys
from os.path import isfile, join
import re

dataset_path = sys.argv[1]

def getSamplesInfo():
	sample_list=[]
	class_info={}
	label_number = 0
	for f in os.listdir(dataset_path):
		if not isfile(join(dataset_path,f)):
			label_number = label_number+1
			for sample in os.listdir(join(dataset_path,f)):
				if not sample.find('-') < 0:
					sample_path = join(join(dataset_path,f),sample)
					group_number = sample.split('-')[0]
					sample_list.append((sample_path,f,label_number,group_number))
	return sample_list

def divideSamples(sample_list, proportion=(0.7,0.1,0.2)):
	total_size = len(sample_metadata)
	traning_size = total_size*proportion[0]
	validation_size = total_size*proportion[1]
	test_size = total_size*proportion[2]
	training_set = sample_list[:traning_size]
	validation_set = sample_list[traning_size+1:traning_size+validation_size]
	test_set = sample_list[traning_size+validation_size+1:]
	return ,validation_set,test_set


def saveSamplesInfo(train_file="train.pkl",val_file="val.pkl",test_file="test.pkl"):
	f_train = open(train_file,'wb')
	f_val = open(val_file,'wb')
	f_test = open(test_file,'wb')
	pickle.dump(getSamplesInfo(), f, pickle.HIGHEST_PROTOCOL)
	f.close()
	print "Metadata saved to ",file

saveSamplesInfo("sample_metadata.pkl")