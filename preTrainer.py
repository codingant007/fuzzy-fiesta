import sys
import timeit
import six.moves.cPickle as pickle

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


from logistic_regression import LogisticRegression, classificationErrors, logisticRegressionTest
from mlp import HiddenLayer, hiddenLayerTest
from conv_pool import ConvPoolLayer, convPoolLayerTest
from sampleLoader import SampleLoader
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
                    minibatch_size=1024):
    
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
    paramDataManager = ParamDataManager(image_shape, channels, nkerns, filter_shapes, hidden_layer, outputs, pools, dropouts, momentum, learning_rate, n_epochs, minibatch_size)
    toLoadParameters = False # Not loading parameters now
    toSaveParameters = True
    paramData = [None]*8
    if toLoadParameters:
        paramData, shapeData = paramDataManager.loadData()
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
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    layer0_input = x.reshape((minibatch_size, channels, image_shape[0], image_shape[0]))


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

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params
    velocity = []
    for i in range(len(params)):
        velocity = velocity + [theano.shared(T.zeros_like(params[i]).eval(), borrow=True)]

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [(velocity_i, momentum * velocity_i + learning_rate * grad_i)
        for velocity_i, grad_i in zip(velocity, grads)]
    updates = updates + [(param_i, param_i - velocity_i)
        for param_i, velocity_i in zip(params, velocity)]

    train_model = theano.function(
        [x,y],
        cost,
        updates=updates,
    )


    ######################
    #     TEST AREA      #
    ######################
    # Test layer 0
    layer0_test_input = x.reshape((minibatch_size, channels, image_shape[0], image_shape[0]))

    # Test layer 0
    layer0_test_output = convPoolLayerTest(
        input=layer0_test_input,
        image_shape=(minibatch_size, channels, image_shape[0], image_shape[0]),
        filter_shape=shapes[0],
        poolsize=(pools[0], pools[0]),
        activation=T.nnet.relu,
        W=layer0.params[0],
        b=layer0.params[1]
    )

    # Test layer 1
    layer1_test_output = convPoolLayerTest(
        input=layer0_test_output,
        image_shape=(minibatch_size, nkerns[0], image_shape[1], image_shape[1]),
        filter_shape=shapes[1],
        poolsize=(pools[1], pools[1]),
        activation=T.nnet.relu,
        W=layer1.params[0],
        b=layer1.params[1]
    )

    # the test HiddenLayer
    layer2_test_input = layer1_test_output.flatten(2)

    # test fully-connected sigmoidal layer
    layer2_test_output = hiddenLayerTest(
        input=layer2_test_input,
        activation=T.nnet.relu,
        W=layer2.params[0],
        b=layer2.params[1]        
    )

    # test the fully-connected sigmoidal layer
    y_pred = logisticRegressionTest(
        input=layer2_test_output,
        W=layer3.params[0],
        b=layer3.params[1]
    )

    # function to validation scores
    validate_model = theano.function(
        [x,y],
        classificationErrors(y_pred, y)
    )

    # create a function to compute test scores
    test_model = theano.function(
        [x,y],
        classificationErrors(y_pred, y)
    )



    #########################
    # TRAIN CONFIGURATION   #
    #########################

    patience = 10000
    patience_increase = 2

    improvement_threshold = 0.995
    momentum_limit = 0.9

    # Initialize training variables
    epoch = 0
    done_looping = False
    minibatch_iteration = 0

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    
    # Initialize sample loader for loading train, val, test samples
    sampleLoader = SampleLoader()

    validation_frequency = min(sampleLoader.n_train_batches,patience//2)
    #validation_frequency = 10
    ############
    # TRAINING #
    ############
    print  "Training ..."

    while (epoch < n_epochs) and (not done_looping):
        #sys.stdout.flush()
        epoch = epoch+1
        learning_rate = learning_rate*0.99
        momentum = momentum + (momentum_limit - momentum)/32
        print('Learning rate = %f, Momentum = %f' %(learning_rate, momentum))

        train_batch_data = sampleLoader.loadNextTrainBatch()
        print train_batch_data[0].shape.eval()

        while train_batch_data is not None:
            train_x,train_y = train_batch_data
            train_x = train_x.get_value()
            train_y = train_y.eval()
            n_minibatches = train_x.shape[0]/minibatch_size
            print type(n_minibatches)
            for minibatch_index in range(n_minibatches):
                minibatch_iteration += 1
                x = train_x[minibatch_index * minibatch_size: (minibatch_index + 1) * minibatch_size].reshape((-1,train_x.shape[-1]))
                y = train_y[minibatch_index * minibatch_size: (minibatch_index + 1) * minibatch_size]
                print "minibatch_iteration ",minibatch_iteration
                cost_minibatch = train_model(x,y)
                print cost_minibatch
                # Validate with a frequency of validation_frequency
                if minibatch_iteration % validation_frequency == 0 :
                    validation_loss = get_validation_loss(sampleLoader,validate_model,minibatch_size)
                    # if we got the best validation score until now
                    print "validation_loss: ",validation_loss," validation_loss: ",best_validation_loss
                    if validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, minibatch_iteration * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = validation_loss
                        best_iter = minibatch_iteration
                        """
                            Check for overfitting logic here
                        """
                        # compute test loss
                        test_loss = get_test_loss(sampleLoader,test_model,minibatch_size)
                        print
                        print "validation loss improved!"
                        print
                        print "validation_loss: ",validation_loss," test_loss: ",test_loss
                        if toSaveParameters:
                            paramDataManager.saveData(params)

                if patience <= minibatch_iteration:
                    done_looping = True
                    break

            train_batch_data = sampleLoader.loadNextTrainBatch()
            

    end_time = timeit.default_timer()
    print "Training complete."
    print "Best Validation Score: ", best_validation_loss," obtained at ",best_iter," With test score ",test_score
    print "Program ran for ",((end_time-start_time)/60),"m"
    return (best_validation_loss,test_score,paramDataManager.getParamDataAddress())

def get_validation_loss(sampleLoader,validate_model,minibatch_size):
    print "n_val_batches: ",sampleLoader.n_val_batches
    val_batch = sampleLoader.loadNextValBatch()
    val_losses = []
    while val_batch is not None :
        x_val,y_val = val_batch
        x_val = x_val.get_value()
        y_val = y_val.eval()
        for i in range(y_val.shape[0]/minibatch_size):
            start_index = i*minibatch_size
            end_index = (i+1)*minibatch_size
            if end_index > y_val.shape[0]:
                break
            x = x_val[start_index:end_index].reshape((-1,x_val.shape[-1]))
            y = y_val[start_index:end_index]
            val_losses += [validate_model(x, y)]
        val_batch = sampleLoader.loadNextValBatch()
    print val_losses
    return np.mean(val_losses)

def get_test_loss(sampleLoader,test_model,minibatch_size):
    test_batch = sampleLoader.loadNextTestBatch()
    test_losses = []
    while test_batch is not None :
        x_test,y_test = test_batch
        x_test = x_test.get_value()
        y_test = y_test.eval()
        for i in range(y_test.shape[0]/minibatch_size):
            start_index = i*minibatch_size
            end_index = (i+1)*minibatch_size
            if end_index > y_test.shape[0]:
                break
            x = x_test[start_index:end_index].reshape((-1,x_test.shape[-1]))
            y = y_test[start_index:end_index]
            test_losses += [test_model(x, y)]
        test_batch = sampleLoader.loadNextTestBatch()
    return np.mean(test_losses)


def experiment(I=0, J=0, K=0, L=0, M=0, N=0):

    nkerns_list = [[50, 100]]
    filter_shapes_list1 = [5]
    filter_shapes_list2 = [5]
    hidden_layer_list = [500]
    dropout_list = [[0.2, 0.4, 0.6]]
    pools_list = [2]

    # nkerns_list = [[8, 16], [12, 24], [16, 32], [24, 48], [32, 64], [48, 96], [64, 128], [96, 256], [192, 256]]
    # filter_shapes_list1 = [3, 5, 7, 9]
    # filter_shapes_list2 = [3, 5, 7, 9]
    # hidden_layer_list = [256, 320, 400, 512, 640, 800, 1024]
    # dropout_list = [[0.07, 0.15, 0.5], [0.1, 0.25, 0.5], [0.2, 0.4, 0.5]]
    # pools_list = [1, 2]

    costs = {}
    for i in range(I, len(nkerns_list)):
        for j in range(J, len(filter_shapes_list1)):
            for k in range(K, len(filter_shapes_list2)):
                for l in range(L, len(hidden_layer_list)):
                    for m in range(M, len(dropout_list)):
                        for n in range(N, len(pools_list)):
                            a = nkerns_list[i]
                            b = filter_shapes_list1[j]
                            c = filter_shapes_list2[k]
                            d = hidden_layer_list[l]
                            e = dropout_list[m]
                            f = pools_list[n]
                            name = str(a + [b] + [c] + [d] + e + [f])
                            print('Working on ' + name)
                            costs[name] = evaluate_cnn(image_shape=[32],
                                            channels=3,
                                            nkerns=a, 
                                            filter_shapes=[b, c],
                                            hidden_layer=[d],
                                            outputs=26,
                                            pools=[f, f],
                                            dropouts=e,
                                            learning_rate=0.001,
                                            momentum=0.5,
                                            n_epochs=2000,
                                            minibatch_size=512)

                            print(costs)
                            output = open('data/costs', 'wb')
                            pickle.dump(costs, output, pickle.HIGHEST_PROTOCOL)
                            output.close()



if __name__ == '__main__':
	experiment()