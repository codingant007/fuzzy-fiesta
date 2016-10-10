import numpy as np
import random
import six.moves.cPickle as pickle
from PIL import Image

class SampleLoader:
	'''
		self.sample_metadata contains samples information
		[
			(sample_1, class_name, class_label, group_number),
			(sample_2, ... ),
			(sample_3, ... ),
			.
			.
			.
		]
	'''

	def _shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')

	def loadNextTrainBatch(self):
		print "loading training batch: ",self.current_train_batch," in epoch: ",self.current_epoch
		train_x = []
		train_y = []
		start_index = self.current_train_batch*self.train_batch_size
		end_index = (self.current_train_batch+1)*self.train_batch_size
		if end_index >= len(self.train_data):
			print "Epoch ",self.current_epoch," completed."
			self.current_epoch += 1
			self.current_train_batch = 0
			shuffleTrainData()
			return

		for i in range(start_index, end_index):
			image = Image.open(self.train_data[i][0])
			train_x.append(np.asarray(image))
			train_y.append(self.train_data[i][2])

		self.current_train_batch += 1

		return _shared_dataset((train_x,train_y))

	def loadNextValBatch(self):
		print "loading validation batch: ",self.current_val_batch," in epoch ",self.current_epoch
		val_x = []
		val_y = []
		start_index = self.current_val_batch*self.train_val_size
		end_index = (self.current_val_batch+1)*self.val_batch_size
		if end_index >= len(self.val_data):
			self.current_val_batch = 0
			shuffleValData()
			return
		
		for i in range(start_index, end_index):
			image = Image.open(self.val_data[i][0])
			val_x.append(np.asarray(image))
			val_y.append(self.val_data[i][2])

		self.current_val_batch += 1

		return _shared_dataset((val_x,val_y))

	def shuffleTrainData(self):
		print "shuffling train dataset for epoch: ",self.current_train_epoch
		random.shuffle(self.train_data)
	def shuffleValData(self):
		print "shuffling val dataset for epoch:",self.current_train_epoch
		random.shuffle(self.val_data)




	def __init__(self,
				 train_batch_size=1024, val_batch_size=100	# Number of samples that are loaded into memory (this is different from mini-batches)
				 train_file="train.pkl", val_file="val.pkl", test_file="test.pkl"):

		print "initializing sample loader ... "

		self.train_data = np.array(pickle.load(open(train_file,'rb')))
		self.val_data = np.array(pickle.load(open(val_data,'rb')))
		self.test_data = np.array(pickle.load(open(test_data,'rb')))

		self.n_train_batches = len(train_data)/train_batch_size
		self.n_val_batches = len(val_data)/val_batch_size
		self.current_train_batch = 0
		self.current_val_batch = 0
		self.current_train_epoch = 0
		print "No of train samples: ",len(train_data)
		print "Number of train batches: ",self.n_train_batches
		print
		print

		shuffleTrainData()
		shuffleValData()

sampleLoader = SampleLoader()











































class RepresentationBalancedSampleLoader:
	'''
		self.sample_metadata contains samples information
		{}
			class1 : [s1,s2,s3....],
			.
			.
			.
		}
	'''


	def generateSampleList(self, number_of_classes_per_batch):
		sample_list=[]

		
		self.sample_list=sample_list


	def __init__(self,
				 batchSize=1000,	# Number of samples that are loaded into memory (this is different from mini-batches)
				 metadata_pkl_file="sample_metadata.pkl"):
		
		print "initializing sample loader ... "
		self.sample_metadata = pickle.load(open(metadata_pkl_file,'rb'))