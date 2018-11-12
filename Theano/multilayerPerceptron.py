""" MULTILAYER PERCEPTRON

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.
"""

# Import dependencies
from __future__ import print_function
__docformat__ = 'restructedtext en'
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T

# Import from Logistic Regression
from logisticRegression import LogisticRegression, load_data

# Import saved model
savedParameters = pickle.load(open('bestMLPModel.pkl'))

#######################################################################################################
##                                       HIDDEN LAYER CLASS
#######################################################################################################

class HiddenLayer(object):

    ## ----------------------------------------------------------------------------------------
    ##                                 The construction function
    ## ----------------------------------------------------------------------------------------
    
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,). NOTE that the nonlinearity
        used here is tanh (i.e. hidden unit activation is given by: tanh(dot(input,W) + b)

        rng (type: numpy.random.RandomState;
             content: a random number generator used to initialize weights)

        input (type: theano.tensor.dmatrix;
               content: a symbolic tensor of shape (n_examples, n_in))

        n_in (type: int;
              content: dimensionality of input)

        n_out (type: int;
               content: number of hidden units)

        activation (type: theano.Op or function;
                    content: Non linearity to be applied in the hidden layer)
        """

        # Link class input with self object
        self.input = input

        ## -------------------------------------------------------------------------------------
        ##  Sets initial values for hidden layer weights
        ## -------------------------------------------------------------------------------------
        """
        `W` is initialized with `W_values` which is uniformely sampled from
        sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden)) for tanh activation function
        
        NOTE that optimal initialization of weights is dependent on the activation
        function used (among other things). For example, results presented in [Xavier10]
        suggest that you should use 4 times larger initial weights for sigmoid
        compared to tanh.
        """

        if W is None:
            # Set weights to result of uniform sampling for the tanh activation function
            W_values = numpy.asarray(
                rng.uniform(
                    low = -numpy.sqrt(6. / (n_in + n_out)),
                    high = numpy.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                # Converted output, using asarray, to dtype theano.config.floatX so that the code is runable on GPU
                dtype = theano.config.floatX
            )

            # If the activation is a sigmoud, the intial weights should be 4 times larger
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            # Set W as the result of this process (i.e. W_values)
            W = theano.shared(value=W_values, name='W', borrow=True)


        ## -------------------------------------------------------------------------------------
        ##  Sets initial values for hidden layer biases
        ## -------------------------------------------------------------------------------------
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        ## -------------------------------------------------------------------------------------
        ##  Get initial output (i.e. input multiplied by weights, plus biases)
        ## -------------------------------------------------------------------------------------
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output) # i.e. apply either tanh or sigmoud function to output activations
        )

        # - Set W and b as the parameters of the model
        self.params = [self.W, self.b]


#######################################################################################################
##                                  MULTILAYER PERCEPTRON CLASS
#######################################################################################################

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        rng (type: numpy.random.RandomState;
             content: a random number generator used to initialize weights)

        input (type: theano.tensor.TensorType;
               content: symbolic variable that describes the input of the architecture (one minibatch))

        n_int (type: int;
               content: number of input units, the dimension of the space in which the datapoints lie)

        n_hidden (type: int;
                  content: number of hidden units)

        n_out (type: int;
               content: number of output units, the dimension of the space in which the labels lie)
        """

        ## -------------------------------------------------------------------------------------
        ##  Define instance of Hidden Layer, as self.hiddenLayer
        ## -------------------------------------------------------------------------------------
        """ Since we are dealing with a one hidden layer MLP, this will translate into a HiddenLayer
        with a tanh activation function connected to the LogisticRegression layer; the activation
        function can be replaced by sigmoid or any other nonlinear function
        """
        self.hiddenLayer = HiddenLayer(
            rng = rng,
            input = input,
            n_in = n_in,
            n_out = n_hidden,
            activation = T.tanh
        )

        ## -------------------------------------------------------------------------------------
        ##  Define instance of Logistic Regression Layer, as self.logRegressionLayer
        ## -------------------------------------------------------------------------------------
        """The logistic regression layer gets as input the hidden units of the hidden layer"""
        self.logRegressionLayer = LogisticRegression(
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_out
        )

        ## -------------------------------------------------------------------------------------
        ##  Regularisation
        ## -------------------------------------------------------------------------------------
        """ We will also use L1 and L2 regularization. L1 and L2 regularization involve adding an
        extra term to the loss function, which penalizes certain parameter configurations. For
        this, we need to compute the L1 norm and the squared L2 norm of the weights.
        """

        # L1 norm (enforcing L1 norm to be small)
        self.L1 = (abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum())

        # Square of L2 norm (enforcing the square of L2 norm to be small)
        self.L2_sqr = ((self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum())


        ## ----------------------------------------------------------------------------------------
        ##                          Calculate loss (negative log likelihood)
        ## ----------------------------------------------------------------------------------------
        
        ## Negative log likelihood of the MLP is given by the negative log likelihood of the output
        ## of the model, computed in the logistic regression layer
        self.negative_log_likelihood = (self.logRegressionLayer.negative_log_likelihood)

        # Same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        ## ----------------------------------------------------------------------------------------
        ##                                  Output parameters
        ## ----------------------------------------------------------------------------------------
        
        # The parameters of the model are the parameters of the two layer it is made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # Keep track of model input
        self.input = input


#######################################################################################################
##                                  MULTILAYER PERCEPTRON CLASS
#######################################################################################################

def perceptronTraining(
    learning_rate = 0.01,
    L1_reg = 0.00, # L1_reg and L2_reg are the hyperparameters controlling the weight of these regularization terms in the total cost function
    L2_reg = 0.0001,
    n_epochs = 1000,
    dataset = 'mnist.pkl.gz',
    batch_size = 20,
    n_hidden = 500):

    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """

    ## --------------------------------------------------------------------------------------
    ##  Load MNIST data (using load_data() [defined above], and the dataset path)
    ## --------------------------------------------------------------------------------------
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0] # devided into training set...
    valid_set_x, valid_set_y = datasets[1] # validation set
    test_set_x, test_set_y = datasets[2] # and test set


    ## --------------------------------------------------------------------------------------
    ##  Compute the number of minibatches for training, validation and testing
    ## --------------------------------------------------------------------------------------
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size # // = division without remainder
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

    # Initialise random number (used to initialise weights)
    rng = numpy.random.RandomState(1234)

    # Construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    # Calulate the cost (negative_log_likelihood + L1_reg + L2_reg)
    """ The cost we minimize during training is the negative log likelihood of the model
    plus the regularization terms (L1 and L2); cost is expressed here symbolically"""
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # Compile a Theano function that computes the mistakes that are made by the model on a minibatch
    # Both for the test model
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    # And for the validation model
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # Compute the gradient of cost with respect to mode parameters
    """ Compute the gradient of cost with respect to theta (sorted in params) the
    resulting gradients will be stored in a list gparams"""
    gparams = [T.grad(cost, param) for param in classifier.params]

    ##  Specify how to update the parameters of the model as a list of (variable, update expression) pairs.
    """zip - Given two lists of the same length, A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4],
    zip generates a list C of same size, where each element is a pair formed from the two lists :
    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]"""
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # Compiling a Theano function `train_model` that returns the cost, but at the same time updates
    # the parameter of the model based on the rules defined in `updates`.
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
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
    done_looping = False
    epoch = 0
    # Loop through epochs
    while (epoch < n_epochs) and (not done_looping): # n_epochs defined in definition of this large function
        epoch = epoch + 1 # Increment epoch on each loop

        # Loop through minibatches
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index) # output current cost
            iter = (epoch - 1) * n_train_batches + minibatch_index # iteration number

            # If the current iter is fully divisible by validation frequency. I.e. concise way of doing something once every 'n' times
            if (iter + 1) % validation_frequency == 0:
                
                # Check for performance (zero-one loss) on validation data set
                validation_losses = [
                    validate_model(i)
                    for i in range(n_valid_batches)
                ]
                this_validation_loss = numpy.mean(validation_losses)

                # Print current validation test results
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
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
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase) # Increase patience

                    # Update best validation loss
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
                          ( epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                          )
                        )

                    # Save the best model (taking the classifier parameters only)
                    with open('bestMLPModel.pkl', 'wb') as f:
                        pickle.dump(classifier.params, f)

            # Else, if patience is expired
            if patience <= iter:
                done_looping = True # Break the loop
                break

    # Now that the loop has ended...
    end_time = timeit.default_timer() # note the time of loop ending

    # Print the ending results
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

# If the current script is the main entrypoint
if __name__ == '__main__':
    perceptronTraining()
