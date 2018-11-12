""" CONVOLUTIONAL MULTI-LAYER PERCEPTRON

This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.

This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling by max.
 - Digit classification is implemented with a logistic regression rather than an RBF network
 - LeNet5 was not fully-connected convolutions at second layer
"""

# Import dependencies
from __future__ import print_function
import six.moves.cPickle as pickle
import gzip
import cPickle
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

# Import from previous scripts
from logisticRegression import LogisticRegression, load_data
from multilayerPerceptron import HiddenLayer

# Import saved model
savedParameters = pickle.load(open('bestCNNModel.pkl'))


#######################################################################################################
##                                 CONVOLUTION AND POOLING LAYER
#######################################################################################################

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    ## ----------------------------------------------------------------------------------------
    ##                                 The construction function
    ## ----------------------------------------------------------------------------------------
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        rng (type: numpy.random.RandomState;
             content: a random number generator used to initialize weights)

        input (type: theano.tensor.dtensor4;
               content: symbolic image tensor, of shape image_shape)

        filter_shape (type: tuple or list of length 4;
               content: (number of filters, num input feature maps, filter height, filter width)

        image_shape (type: tuple or list of length 4;
               content: (batch size, num input feature maps, image height, image width)

        poolsize (type: tuple or list of length 2
               content: the downsampling (pooling) factor (#rows, #cols))
        """

        # Make sure that there are the same number of input feature maps in input_shape and filter_shape
        assert image_shape[1] == filter_shape[1]
        
        # Assign input to self object
        self.input = input

        # Define the number of inputs to each hidden unit
        """there are "num input feature maps * filter height * filter width" inputs (i.e. filter_shape[1:]"""
        fan_in = numpy.prod(filter_shape[1:])

        # Define the number of gradients recieved by each unit in the lower layer
        """num output feature maps * filter height * filter width" / pooling size"""
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) // numpy.prod(poolsize))

        ## -------------------------------------------------------------------------------------
        ##  Initialize weights (for filters) with random values
        ## -------------------------------------------------------------------------------------    
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(
                    low = -numpy.sqrt(6. / (fan_in + fan_out)), # Same as in MLP
                    high = numpy.sqrt(6. / (fan_in + fan_out)),
                    size = filter_shape
                ),
                dtype = theano.config.floatX
            )
        )

        # Set the biases is a 1D tensor (i.e. one bias per output feature map)
        b_values = numpy.zeros((filter_shape[0],), dtype = theano.config.floatX)
        self.b = theano.shared(value = b_values, borrow = True)

        ## -------------------------------------------------------------------------------------
        ##  Define convolution (i.e. convolve input feature maps with filters)
        ## -------------------------------------------------------------------------------------
        conv_out = conv2d(
            input = input,
            filters = self.W,
            filter_shape = filter_shape,
            input_shape = image_shape
        )

        ## -------------------------------------------------------------------------------------
        ##  Define max-pooling (pool each feature map individually, using maxpooling)
        ## -------------------------------------------------------------------------------------
        """ Max-pooling is a form of non-linear down-sampling. Max- pooling partitions
        the input image into a set of non-overlapping rectangles and, for each such
        sub-region, outputs the maximum value. By eliminating non-maximal values, it
        reduces computation for upper layers"""
        pooled_out = pool.pool_2d(
            input = conv_out,
            ws = poolsize,
            ignore_border = True
        )

        # Add the bias term.
        """ Since the bias is a vector (1D array), we first reshape it to a tensor
        of shape (1, n_filters, 1, 1). Each bias will thus be broadcasted across
        mini-batches and feature map (width & height)"""
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # Store parameters of this layer
        self.params = [self.W, self.b]

        # Keep track of model input
        self.input = input


##########################################################################################################
##                         BUILD AND TRAIN CONVOLUTIONAL NEURAL NETWORK
##########################################################################################################     

def evaluate_lenet5(
    learning_rate = 0.1,
    n_epochs = 200,
    dataset = 'mnist.pkl.gz',
    nkerns = [20, 50],
    batch_size = 500):

    """
    learning_rate (type: float;
                content: learning rate used (factor for the stochastic gradient)

    n_epochs (type: int;
             content: maximal number of epochs to run the optimizer)

    dataset (type: string;
            content: path to the dataset used for training /testing (MNIST here))

    nkerns (type: list of ints;
            content: number of kernels on each layer
    """

    # Initialise random number (used to initialise weights)
    rng = numpy.random.RandomState(23455)

    ## --------------------------------------------------------------------------------------
    ##  Load MNIST data (using load_data() [defined above], and the dataset path)
    ## --------------------------------------------------------------------------------------
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0] # devided into training set...
    valid_set_x, valid_set_y = datasets[1] # validation set
    test_set_x, test_set_y = datasets[2] # and test set

    # Compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


    #########################################################################################
    #                                    BUILD THE MODEL                                    #
    #########################################################################################
    print('... building the model')

    # Allocate (initialise) symbolic variables and generate symbolic variables for input (x and y represent a minibatch)
    index = T.lscalar()  # index to a [mini]batch (lscalar() returns a zero-dimension value)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    
    ## --------------------------------------------------------------------------------------
    ##  Define the FIRST layer
    ## --------------------------------------------------------------------------------------
    
    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28) to a 4D tensor,
    # compatible with our LeNetConvPoolLayer. (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input = layer0_input,
        image_shape = (batch_size, 1, 28, 28),
        filter_shape = (nkerns[0], 1, 5, 5),
        poolsize = (2, 2)
    )

    ## --------------------------------------------------------------------------------------
    ##  Define the SECOND layer
    ## --------------------------------------------------------------------------------------

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    ## --------------------------------------------------------------------------------------
    ##  Define the THIRD layer
    ## --------------------------------------------------------------------------------------

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    ## --------------------------------------------------------------------------------------
    ##  Define the FOURTH layer
    ## --------------------------------------------------------------------------------------

    # Classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)


    ## --------------------------------------------------------------------------------------
    ##  Define cost and test functions
    ## --------------------------------------------------------------------------------------
    cost = layer3.negative_log_likelihood(y) # Calulate the cost (negative_log_likelihood)

    # Compile a Theano function that computes the mistakes that are made by the model on a minibatch
    # Both for the test model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # And for the validation model
    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # Create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    ##  Specify how to update the parameters of the model
    """ train_model is a function that updates the model parameters by SGD.
    Since this model has many parameters, it would be tedious to manually
    create an update rule for each model parameter. We thus create the
    updates list by automatically looping over all (params[i], grads[i]) pairs.
    """
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    # Compile a Theano function `train_model` that returns the cost, but at the same time updates
    # the parameter of the model based on the rules defined in `updates`.
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    #########################################################################################
    #                                       TRAIN MODEL                                     #
    #########################################################################################
    print('... training the model')

    ## --------------------------------------------------------------------------------------
    ##  Define early-stopping parameters
    ## --------------------------------------------------------------------------------------
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many minibatches before checking the network
                                  # on the validation set; in this case we check every epoch
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    ## --------------------------------------------------------------------------------------
    ##  Start iterating loop (i.e. through multibatches for repeated SGD)
    ## --------------------------------------------------------------------------------------
    epoch = 0
    done_looping = False
    # Loop through epochs
    while (epoch < n_epochs) and (not done_looping): # n_epochs defined in definition of this large function
        epoch = epoch + 1 # Increment epoch on each loop

        # Loop through minibatches
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index # iteration number

            ## On every 100 iterations...
            if iter % 100 == 0:
                print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)

            # When the iteration is fully divisible by the validation frequency
            if (iter + 1) % validation_frequency == 0:

                # Check for performance (zero-one loss) on validation data set
                validation_losses = [
                    validate_model(i)
                    for i in range(n_valid_batches)
                ]
                this_validation_loss = numpy.mean(validation_losses)

                # Print current validation test results
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (
                          epoch,
                          minibatch_index + 1,
                          n_train_batches,
                          this_validation_loss * 100.
                      )
                )

                # If we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # ...and if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # Save the best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # Test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)

                    # Print test results
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    ## -----------------------------------------------------------------
                    ##  Save model parameters using cPickle
                    ## -----------------------------------------------------------------
                    fname = 'bestCNNModel.pkl'
                    saveFile = open(fname, 'wb')

                    # model weights
                    cPickle.dump(layer0.W, saveFile)
                    cPickle.dump(layer0.b, saveFile)
                    cPickle.dump(layer1.W, saveFile)
                    cPickle.dump(layer1.b, saveFile)
                    cPickle.dump(layer2.W, saveFile)
                    cPickle.dump(layer2.b, saveFile)

                    """
                    # hyperparameters and performance
                    cPickle.dump(learning_rate, saveFile)
                    cPickle.dump(best_validation_loss, saveFile)
                    cPickle.dump(test_score, saveFile)
                    cPickle.dump(test_losses, saveFile)
                    cPickle.dump(nkerns, saveFile)
                    cPickle.dump(n_epochs, saveFile)
                    cPickle.dump(batch_size, saveFile)
                    """
                    saveFile.close()

            # Else, if patience is expired
            if patience <= iter:
                done_looping = True # Break the loop
                break

    # Now that the loop has ended...
    end_time = timeit.default_timer() # note the time of loop ending

    # Print the ending results
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

# If the current script is the main entrypoint
if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
