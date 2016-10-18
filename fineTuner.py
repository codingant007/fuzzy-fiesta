import sys

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


from logistic_regression import LogisticRegression, classificationErrors, logisticRegressionTest
from mlp import HiddenLayer, hiddenLayerTest
from conv_pool import ConvPoolLayer, convPoolLayerTest
from groupLoader import GroupLoader
from paramDataManager import ParamDataManager


def evaluate_cnn(image_shape=[32],
                    channels=3,
                    nkerns=[64, 128], 
                    filter_shapes=[5, 5],
                    hidden_layer=[1024],
                    outputs=10,
                    pools=[2, 2],
                    dropouts=[0.1, 0.25, 0.5],
                    learning_rate=0.1,
                    momentum=0.5,
                    n_epochs=2000,
                    minibatch_size=64):
	
	rng = np.random.RandomState(12345)

    # calculate shapes at each CNN layer
    for i in range(len(filter_shapes)):
        if (image_shape[-1] - filter_shapes[i] + 1) % pools[i] != 0 :
            return -1
        image_shape = image_shape + [(image_shape[-1] - filter_shapes[i] + 1) // pools[i]]

    # specify shape of filters
    shapes = [(nkerns[0], channels, filter_shapes[0], filter_shapes[0]), 
                (nkerns[1], nkerns[0], filter_shapes[1], filter_shapes[1]), 
                (nkerns[1] * image_shape[-1]**2 , hidden_layer[0]), 
                (hidden_layer[0], outputs)]


    # load parameters
    paramLoader = ParamDataManager(image_shape, channels, nkerns, filter_shapes, hidden_layer, outputs, pools, dropouts, momentum, learning_rate, n_epochs, minibatch_size) 
    paramLoader.loadData
    toLoadParameters = True # Loading paramters tuned in preTrainer
    toSaveParameters = True
    paramData = [None]*8
    if toLoadParameters:
        paramData, shapeData = paramLoader.loadData(paramDataAddress)
        shapeMatched = True
        for i in range(len(shapes)):
            if(shapes[-i-1] != shapeData[2*i]):
                paramData[2*i] = None
                paramData[2*i + 1] = None
                print(".. Shape problem for %d .." % (2*i), shapes[-i], shapeData[2*i])
                shapeMatched = False
            else:
                print('... Data loaded for layer %d ...' % i)
        if(shapeMatched == False):
            print('... Shape did not match ...')



    #######################
    # Variables for model #
    #######################

    x = T.matrix('x')
    y = T.ivector('y')


    ######################
    #     TRAIN AREA     #
    ######################
    # Construct the first convolutional pooling layer:
    layer0 = ConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(minibatch_size, channels, image_shape[0], image_shape[0]),
        filter_shape=shapes[0],
        poolsize=(pools[0], pools[0]),
        activation=T.nnet.relu,
        dropout=dropouts[0],
        W=paramData[6],
        b=paramData[7]
    )

    # Construct the second convolutional pooling layer
    layer1 = ConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(minibatch_size, nkerns[0], image_shape[1], image_shape[1]),
        filter_shape=shapes[1],
        poolsize=(pools[1], pools[1]),
        activation=T.nnet.relu,
        dropout=dropouts[1],
        W=paramData[4],
        b=paramData[5]
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of

    layer2_input = layer1.output.flatten(2)     # shape = (7*7*128 , 64)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=shapes[2][0],
        n_out=shapes[2][1],
        activation=T.nnet.relu,
        dropout=dropouts[2],
        W=paramData[2],
        b=paramData[3]        
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(
        input=layer2.output,
        n_in=shapes[3][0],
        n_out=shapes[3][1],
        W=paramData[0],
        b=paramData[1]
    )

    groupLoader = GroupLoader()


    