import numpy as np
import random
import six.moves.cPickle as pickle
from PIL import Image

class GroupLoader:
	'''
		tain_group.pkl, val_group.pkl, test_group.pkl contain sample file information in the array as
		[
			(group_label1, class, class_label, [sample1-1,sample1-2, ...sample1-150]),
			...
			...
		]
	'''


	def __init__(self, \
					train_batch_size=1024, val_batch_size=100, test_batch_size=100, \
					train_filename="train.pkl", val_filename="val.pkl", test_filename="test.pkl"):

		print "initializing sample loader ... "

		self.train_data = np.array(pickle.load(open(train_filename,'rb')))
		self.val_data = np.array(pickle.load(open(val_filename,'rb')))
		self.test_data = np.array(pickle.load(open(test_filename,'rb')))

		self.n_train_batches = len(train_data)/train_batch_size
		self.n_val_batches = len(val_data)/val_batch_size
		self.n_test_batches = len(test_data)/test_batch_size

		self.current_train_batch = 0
		self.current_val_batch = 0
		self.current_test_batch = 0

		self.current_train_epoch = 0
		
		print "No of train samples: ",len(train_data)
		print "Number of train batches: ",self.n_train_batches
		print
		print

		shuffleTrainData()
		shuffleValData()

		sampleLoader = SampleLoader()


	