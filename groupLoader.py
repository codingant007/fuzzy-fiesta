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

	def loadNextTrainBatch(self):
		print "loading training batch: ",self.current_train_batch," in epoch: ",self.current_train_epoch
		train_x = []
		train_y = []
		start_index = self.current_train_batch*self.train_batch_size
		end_index = (self.current_train_batch+1)*self.train_batch_size
		if end_index >= len(self.train_data):
			print "Epoch ",self.current_train_epoch," completed."
			self.current_train_epoch += 1
			self.current_train_batch = 0
			self.shuffleTrainData()
			return

		for i in range(start_index, end_index):
			image_mat = np.asarray( Image.open(self.train_data[i][0]))
			# There some greyscale imgaes in the dataset repeat in 3rd dim to deal with this
			if image_mat.ndim == 2:
				image_mat = image_mat.reshape((image_mat.shape[0],image_mat.shape[0],1))
				image_mat = np.repeat(image_mat,3,axis=2)
			train_x.append(image_mat)
			train_y.append(self.train_data[i][2])

		self.current_train_batch += 1

		train_x = np.asarray(train_x,dtype=theano.config.floatX)
		train_y = np.asarray(train_y,dtype=theano.config.floatX)

		assert train_x.shape == (self.train_batch_size,32,32,3)
		assert train_y.shape == (self.train_batch_size,)

		return _shared_dataset((train_x,train_y))

	def loadNextValBatch(self):
		print "loading validation batch: ",self.current_val_batch," in epoch ",self.current_train_epoch
		val_x = []
		val_y = []
		start_index = self.current_val_batch*self.val_batch_size
		end_index = (self.current_val_batch+1)*self.val_batch_size
		if end_index >= len(self.val_data):
			self.current_val_batch = 0
			self.shuffleValData()
			return
		
		for i in range(start_index, end_index):
			image_mat = np.asarray( Image.open(self.val_data[i][0]))
			# There some greyscale imgaes in the dataset repeat in 3rd dim to deal with this
			if image_mat.ndim == 2:
				image_mat = image_mat.reshape((image_mat.shape[0],image_mat.shape[0],1))
				image_mat = np.repeat(image_mat,3,axis=2)
			val_x.append(image_mat)
			val_y.append(self.val_data[i][2])

		self.current_val_batch += 1

		val_x = np.asarray(val_x,dtype=theano.config.floatX)
		val_y = np.asarray(val_y,dtype=theano.config.floatX)

		assert val_x.shape == (self.val_batch_size,32,32,3)
		assert val_y.shape == (self.val_batch_size,)

		return _shared_dataset((val_x,val_y))

	def loadNextTestBatch(self):
		print "loading Test batch: ",self.current_test_batch," in epoch ",self.current_train_epoch
		test_x = []
		test_y = []
		start_index = self.current_test_batch*self.test_batch_size
		end_index = (self.current_test_batch+1)*self.test_batch_size
		if end_index >= len(self.test_data):
			self.current_test_batch = 0
			self.shuffleTestData()
			return
		
		for i in range(start_index, end_index):
			image_mat = np.asarray( Image.open(self.test_data[i][0]))
			# There some greyscale imgaes in the dataset repeat in 3rd dim to deal with this
			if image_mat.ndim == 2:
				image_mat = image_mat.reshape((image_mat.shape[0],image_mat.shape[0],1))
				image_mat = np.repeat(image_mat,3,axis=2)
			test_x.append(image_mat)
			test_y.append(self.test_data[i][2])

		self.current_test_batch += 1

		test_x = np.asarray(test_x,dtype=theano.config.floatX)
		test_y = np.asarray(test_y,dtype=theano.config.floatX)

		assert test_x.shape == (self.test_batch_size,32,32,3)
		assert test_y.shape == (self.test_batch_size,)

		return _shared_dataset((test_x,test_y))


	def shuffleTrainData(self):
		print "shuffling train dataset for epoch: ",self.current_train_epoch
		random.shuffle(self.train_data)
	def shuffleValData(self):
		print "shuffling val dataset for epoch:",self.current_train_epoch
		random.shuffle(self.val_data)
	def shuffleTestData(self):
		print "shuffling test dataset for epoch:",self.current_train_epoch
		random.shuffle(self.test_data)


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


	