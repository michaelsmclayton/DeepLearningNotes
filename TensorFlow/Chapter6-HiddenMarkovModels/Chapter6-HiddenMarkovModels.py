# Markov Models (MMs)
''' Markov models describe the probability of transitions betweeen
different states. An HMM describes the likeihood of each of these state transitions
at a single moment in time. A robot that decides which action to perform based only
on its current state is said to follow the MARKOV PROPERTY.

    States and their transitions can be drawn as directed graphs with nodes and
edges. Each edge has a weight representing a probability. A more efficient way to
represent MMs is with a matrix known as a TRANSITION MATRIX. Here, if you have N
states, the transition matrix will be N * N in size. Given that nodes in a
TensorFlow graph are Tensors, we can represent transition matrices as nodes in
TensorFlow. We can then apply mathematical operators on these nodes to achieve
interesting results.
'''

# Hidden Markov Models (HMMs)
''' The Markov model defined in the previous section is convenient when all the
states are observable, but that is not always the case. Sometimes there are only
a few things that we can measure (e.g. age, blood tests, etc), and we must use
these to infer states that are not directly observable (e.g. cancer). The assumption
is that unobservable states will leave traces of themselves. To describe the likelihood
of observing these traces during a given hidden state, we use an EMISSION MATRIX.
The number of rows of the matrix is the number of states (e.g. Sunny, Cloudy, Rainy), and
the number of columns is the number of different types of observations (e.g. Hot, Mild,
 Cold). In addition to the transition matrices that are used in simpler MMs, we also
use an INITIAL PROBABILITIES VECTOR to describe the likelihood of a given state
with no prior information (e.g. it is generally more likely to be sunny in LA vs.
London).
'''

'''Here, we use the FORWARD ALGORITHM to calculate the probability of a given series
of observations'''

# Import the usual libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np


class HMM(object):
    def __init__(self, initialProbabilities, transitionalProbabilities, emissionProbabilities):
        self.N = np.size(initialProbabilities)
        self.initialProbabilities = initialProbabilities # Initial probabilities of each state
        self.transitionalProbabilities = transitionalProbabilities # Transition probability matrix 
        self.emissionProbabilities = tf.constant(emissionProbabilities) # Emission probability matrix

        # Double-check the shapes of all the matrices makes sense
        assert self.initialProbabilities.shape == (self.N, 1)
        assert self.transitionalProbabilities.shape == (self.N, self.N)
        assert emissionProbabilities.shape[0] == self.N

        # Define the placeholders used for the forward algorithm
        self.observationID = tf.placeholder(tf.int32) # Observation IDs
        self.cache = tf.placeholder(tf.float64)

    # Get emission probabilities for current observation (i.e. likelihood of current observation for each hidden state)
    def get_emission(self, observationID): # Function to access a column from the emission matrix (just a helper function to efficiently obtain data from an arbitrary matrix)
        slice_location = [0, observationID] # The location of where to slice the emission matrix
        num_rows = tf.shape(self.emissionProbabilities)[0] # Number of rows in emission matrix (observation * observation probabilities matrix)
        slice_shape = [num_rows, 1] # The shape of the slice
        currentEmissionProbabilities = tf.slice(self.emissionProbabilities, slice_location, slice_shape) # Perform the slicing operator [ slice(input_, begin, size, name=None) ]
        return currentEmissionProbabilities
    ''' Essentially, this function slices the observation*observation probabilities matrix. The
    number of rows in this matrix is the number of hidden states (e.g. Sunny, Cloudy, Rainy), and
    the number of columns is the number of different types of observations (e.g. Hot, Mild, Cold).
    This gives a vector with the emission probabilities for all the hidden states, given the current
    observation.
    '''

    # Define initialising operation to multiply initial probabilities with extracted row from emission probability
    def forward_init_op(self):
        currentEmissionProbabilities = self.get_emission(self.observationID)
        forwardOperation = tf.multiply(self.initialProbabilities, currentEmissionProbabilities)
        return forwardOperation # for a given observation, what are the emission probabilities for each state, multiplied by the initial probabilities of each state

    # Update the cache at each observation
    def forward_op(self):
        # Get the current emission probabilities
        currentEmissionProbabilities = tf.transpose(self.get_emission(self.observationID))

        # Multiply the old cache (with current emission probabilities)
        newCacheGivenCurrentEmissionProbs = tf.matmul(self.cache, currentEmissionProbabilities)

        # Weight current cache by probability of next state (i.e. bias cache by liklihood of state)
        '''Rather than multiply by the initial probabilities, you multiply by the transition matrix
        and then sum up the rows to weight the next cache by the probability of moving to state 1 vs. 2'''
        cacheWeightedByTransitions = newCacheGivenCurrentEmissionProbs * self.transitionalProbabilities

        # Get probability of observation given any states (i.e. sum of all probabilities of current observation)
        cache = tf.reduce_sum(cacheWeightedByTransitions, 0)
        return tf.reshape(cache, tf.shape(self.cache))


def forward_algorithm(sess, hmm, observations):
    # Initialise vector with initial probabilities * emission probabilities for the first observation
    cache = sess.run(hmm.forward_init_op(), feed_dict={hmm.observationID: observations[0]})
    
    # Loop over observations
    for t in range(1, len(observations)):
        cache = sess.run(hmm.forward_op(), feed_dict={hmm.observationID: observations[t], hmm.cache: cache}) # Update cache
    prob = sess.run(tf.reduce_sum(cache)) # 
    return prob

if __name__ == '__main__':
    initialProbabilities = np.array([[0.6], [0.4]]) # Two hidden states
    transitionalProbabilities = np.array([[0.7, 0.3], [0.4, 0.6]]) # Transitions from state 1-1, 1-2, 2-1, 2-2
    emissionProbabilities = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]]) # Emission probabilities for each observation (column), given a certain hidden state (row)
    # Initial probabilities     Transitional probabilities       Emmision probabilities 
    #     (State x 1)                (State x State)               (State x Observation)
    #     [   0.6,                  [   0.7, 0.3,                 [   0.5, 0.4, 0.1
    #         0.4   ]                   0.4, 0.6    ]                 0.1, 0.3, 0.6    ]

    hmm = HMM(initialProbabilities=initialProbabilities, transitionalProbabilities=transitionalProbabilities, emissionProbabilities=emissionProbabilities)

    # 0.6 * 0.5 = 0.3       0.6 * 0.4 = .24         0.6 * 0.1 = .06
    # 0.4 * 0.1 = 0.04      0.4 * 0.3 = .12         0.4 * 0.6 = .24
    #           = .34                 = .36                   = .30   = 1

    # .34 + .36 + .3
    observations = [0,1] # [0, 1, 1, 2, 1]
    with tf.Session() as sess:
        prob = forward_algorithm(sess, hmm, observations)
        print('Probability of observing {} is {}'.format(observations, prob))

# Initials
trans = np.array([[.7,.3],[.4,.6]])
initialProbs = np.array([[.5],[.1]])

# First step
startingEmission = np.array([[.5],[.1]]) # observation 0
fwd = startingEmission * initialProbs * trans
fwd = np.sum(fwd,0)
fwd = fwd.reshape((-1,1))

# Second step
secondEmission = np.array([[.4],[.3]]) # observation 0
fwd = fwd * secondEmission * trans
fwd = np.sum(fwd,0)
fwd = fwd.reshape((-1,1))

fwd = np.sum(fwd,0)
print(fwd)

