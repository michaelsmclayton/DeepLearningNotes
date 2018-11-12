""" LOGISTIC REGRESSION

    Logistic regression is a probabilistic, linear classifier. It is parametrized
    by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
    done by projecting data points onto a set of hyperplanes, the distance to
    which is used to determine a class membership probability.
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


#######################################################################################################
##                                       LOGISTIC REGRESSION CLASS
#######################################################################################################

class LogisticRegression(object):

    ## ----------------------------------------------------------------------------------------
    ##                                 The construction function
    ## ----------------------------------------------------------------------------------------
    
    def __init__(self, input, n_in, n_out): ## Note that "__init__" marks a constructor function in Python

        """ input (type: theano.tensor.TensorType;
                    content: symbolic variable that describes the input of the architecture (one minibatch)
            n_in (type: int;
                  content: number of input units, the dimension of the space in which the datapoints lie)
            n_out (type: int;
                  content:number of output units, the dimension of the space in which the labels lie
        """

        ## -------------------------------------------------------------------------------------
        ##  Sets initial values for logistic regression weights and biases.
        ## -------------------------------------------------------------------------------------

        # - 1) Weights (initialise with a matrix of zeroes, with a shape of [n_in, n_out])
        self.W = theano.shared( # For memory-efficient data storange
            value = numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
            name = 'W', # names is 'W', for weights
            borrow = True # Something about always returning the original data and not a copy?
        )

        # - 2) Biases (initialise with a vector of zeros, with length = n_out)
        self.b = theano.shared(
            value = numpy.zeros((n_out,), dtype = theano.config.floatX),
            name = 'b',
            borrow = True
        )

        # Use parameters of best model (if saved)
        if __name__ == '__main__': # (only when the current script is the main entrypoint; i.e. not when it is imported by other scripts)
            try:
                savedClassifier = pickle.load(open('bestLogisticRegressionModel.pkl'))
                self.W = savedClassifier.W
                self.b = savedClassifier.b
            except:
                print('No best model found. Using random weights and biases')

        # - Set W and b as the parameters of the model
        self.params = [self.W, self.b]

        # - Keep track of model input
        self.input = input

        ## --------------------------------------------------------------------------------------
        ##  Calculate intial probability matrix of class-membership (i.e. with weights and biases at zero)
        ## --------------------------------------------------------------------------------------
        # W is a matrix where column-k represent the separation hyperplane for class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyperplane-k

        # T.dot(input, self.W) + self.b = linear regression output (i.e. input*weight + bias)
        # T.nnet.softmax() = sigmoud function
        # y = class membership; x = input data
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)


        ## --------------------------------------------------------------------------------------
        ##  Find which prediction of class, for each input example, which has the heighest probability
        ## --------------------------------------------------------------------------------------
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


    ## ----------------------------------------------------------------------------------------
    ##                          Calculate loss (negative log likelihood)
    ## ----------------------------------------------------------------------------------------
    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction of
        this model under a given target distribution. Note: we use the mean instead
        of the sum so that the learning rate is less dependent on the batch size

        y (type: theano.tensor.TensorType;
           content: corresponds to a vector that gives the correct label for each example)

        y.shape[0] is (symbolically) the number of rows in y, i.e., number of examples
        (call it n) in the minibatch.
        
        T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        
        T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP) with one row
        per example and one column per class
        
        LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]], LP[1,y[1]],
        LP[2,y[2]], ..., LP[n-1,y[n-1]]]
        
        T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples) of the
        elements in v(y?), i.e., the mean log-likelihood across the minibatch.
        """

        ## Estimate loss
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    ## ----------------------------------------------------------------------------------------
    ##                         Calculate number of errors (zero one loss)
    ## ----------------------------------------------------------------------------------------
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        y (type: theano.tensor.TensorType;
           content: corresponds to a vector that gives the correct label for each example)
        """

        # Make sure that the vector of correct labels has the same dimensions as the predicted labels
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))

        # Make sure that y (i.e. the vector of correct labels) is of the correct datatype
        if y.dtype.startswith('int'):
            # T.neq returns a vector of 0s and 1s, where 1 represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


#####################################################################################################
##                                      LOAD DATA FUNCTION
#####################################################################################################   

def load_data(dataset):
    '''
    dataset (type: string;
             content: the path to the dataset (here MNIST))
    '''
    ## --------------------------------------------------------------------------------------
    ##  Download the MNIST dataset if it is not present
    ## --------------------------------------------------------------------------------------
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)


    ## --------------------------------------------------------------------------------------
    ##  Load the dataset
    ## --------------------------------------------------------------------------------------
    print('... loading data')
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    
    # train_set, valid_set, test_set format: tuple(input, target)
    # where...
    # input: a numpy.ndarray of 2 dimensions (a matrix) where each row corresponds to an example.
    # target: a numpy.ndarray of 1 dimension (vector) that has the same length as the number of
    #         rows in the input. It should give the target to the example with the same index in
    #         the input.


    ## --------------------------------------------------------------------------------------
    ##  Load dataset as shared data
    ## --------------------------------------------------------------------------------------
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                 dtype = theano.config.floatX),
                                 borrow = borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                 dtype = theano.config.floatX),
                                 borrow = borrow)

        # When storing data on the GPU, it has to be stored as floats.
        # Therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations,
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense). Therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    ## --------------------------------------------------------------------------------------
    ##  Load shared data (using the above defined function)
    ## --------------------------------------------------------------------------------------
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval # Return all data together


##########################################################################################################
##                               STOCHASTIC GRADIENT DESCENT OPTIMISATION
##########################################################################################################     

def sgd_optimization_mnist(
    learning_rate=0.13,
    n_epochs=1000,
    dataset='mnist.pkl.gz',
    batch_size=600):
    
    """
    Demonstrate stochastic gradient descent optimization of a log-linear model (demonstrated on MNIST)

    learning_rate (type: float;
                   content: learning rate used (factor for the stochastic gradient)

    n_epochs (type: int;
              content: maximal number of epochs to run the optimizer)

    dataset (type: string;
             content: the path of the MNIST dataset file from http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz)
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

    # Construct the logistic regression class
    classifier = LogisticRegression(input = x, n_in= 28*28, n_out = 10) # Each MNIST image has size 28*28

    # Define the cost we minimize during training (i.e. the negative log likelihood of the model) in symbolic format
    cost = classifier.negative_log_likelihood(y) # input y is the true, labelled classes for training

    # Compile a Theano function that computes the mistakes that are made by the model on a minibatch
    # Both for the test model
    test_model = theano.function(
        inputs = [index], # index of the current minibatch
        outputs = classifier.errors(y), # calculate current cost (zero one error)
        givens = {
            x: test_set_x[index*batch_size : (index+1)*batch_size], # Given the current range of minbatch index values
            y: test_set_y[index*batch_size : (index+1)*batch_size]
        }
    )
    # And for the validation model
    validate_model = theano.function(
        inputs = [index],
        outputs = classifier.errors(y), # i.e. return the current cost
        givens = {
            x: valid_set_x[index*batch_size : (index+1)*batch_size],
            y: valid_set_y[index*batch_size : (index+1)*batch_size]
        }
    )

    ## --------------------------------------------------------------------------------------
    ##  Compute the gradient of cost with respect to theta = (W,b) (i.e. given the paremeters of W and b)
    ## --------------------------------------------------------------------------------------
    gradient_W = T.grad(cost = cost, wrt = classifier.W) # gradient for weights
    gradient_b = T.grad(cost = cost, wrt = classifier.b) # gradient for biases

    ## --------------------------------------------------------------------------------------
    ##  Specify how to update the parameters of the model as a list of (variable, update expression) pairs.
    ## --------------------------------------------------------------------------------------
    """ For a parameter to change (e.g. weight) the current value is minused the current gradient multiplied
    by the learning rate (i.e. direction * magnitude of a vector). This rule for updating is stored in a list
    of (variable, update expression) pairs. I.e. the first element is the symbolic variable to be updated in
    the step, and the second element is the symbolic function for calculating its new value.
    """
    updates = [(classifier.W, classifier.W - learning_rate*gradient_W),
               (classifier.b, classifier.b - learning_rate*gradient_b)]

    
    ## --------------------------------------------------------------------------------------
    # Compiling a Theano function `train_model` that returns the cost, but at the same time updates
    # the parameter of the model based on the rules defined in `updates`.
    ## --------------------------------------------------------------------------------------
    """ Each time train_model(index) is called, it will thus compute and return the cost of a minibatch,
    while also performing a step of MSGD. The entire learning algorithm thus consists in looping over all
    examples in the dataset, considering all the examples in one minibatch at a time, and repeatedly
    calling the train_model function.
    """
    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
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
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many minibatches before checking the network
                                  # on the validation set; in this case we check every epoch
    best_validation_loss = numpy.inf
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
                    validate_model(i)  # compute zero-one loss on validation set
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
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                       patience = max(patience, iter * patience_increase) # Increase patience

                    # Update best validation loss
                    best_validation_loss = this_validation_loss
                    
                    # Test it on the test set
                    test_losses = [test_model(i)
                                   for i in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    # Print test results
                    print(
                        (' epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (   epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # Save the best model
                    with open('bestLogisticRegressionModel.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            # Else, if patience is expired
            if patience <= iter:
                done_looping = True # Break the loop
                break

    # Now that the loop has ended...
    end_time = timeit.default_timer() # note the time of loop ending

    # Print the ending results
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)



##########################################################################################################
##                               FUNCTION TO PREDICT LABELS
##########################################################################################################     
def predict():
    """
    An example of how to load a trained model and use it to predict labels.
    """

    # Load the saved model (created at the end of sgd_optimization_mnist())
    classifier = pickle.load(open('bestLogisticRegressionModel.pkl'))

    # Compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test set
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()

    numberOfPredictions = 40

    # Print correct labels
    print(' ')
    correctLabels = T.cast(test_set_y[:numberOfPredictions], 'int32')
    print("Correct values for the first 15 examples in test set:")
    print(correctLabels.eval())

    # Print predicted labels
    print(' ')
    predictedLabels = predict_model(test_set_x[:numberOfPredictions])
    print("Predicted values for the first 15 examples in test set:")
    print(predictedLabels)


# If the current script is the main entrypoint
if __name__ == '__main__':
    sgd_optimization_mnist()
    predict()
