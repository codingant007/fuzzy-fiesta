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

	