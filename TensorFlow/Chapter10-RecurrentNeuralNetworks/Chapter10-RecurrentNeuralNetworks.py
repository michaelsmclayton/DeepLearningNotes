''' Recurrent neural networks use context to answer questions. An RNN with a
simple structure will take as input a vector X(t) and generate an output a
vector Y(t), at some time (t). The middle of this interaction is a hidden
neuron. In other words, the network has a 'W_in' matrix (dealing with input
to hidden neuron flow), and a 'W_out' matrix (dealing with hidden neuron to
output flow). When looking at a sequence of data, we can arrange these
structures into a row, with each column (input, hidden, output) processing
a different time, t. However, without connections between these columns,
processing at one time, t, won't be able to take previous data into account
(i.e. from t-1). Consequently, we add a new transition weight matrix to the
network, called 'W_transmission' (only called 'W' in the book). This gives
a weight from each neuron at time, t, to itself at time t+1. The introduction
of the transition weight means that the next state is now dependent on the
previous model, as well as the previous state. This means that our model has
a "memory" of what previously happened.
'''

# Implementing a recurrent neural network
'''  One type of RNN model is called Long Short-Term Memory (LSTM). It means
exactly what it sounds like: short-term patterns aren't forgotten in the
long-term. The precise implementation detail of LSTM is not in the scope of
this book. There is no definite standard yet. That is where TensorFlow comes
in. It takes care of how the model is defined so you can use it out-of-the-box.
It also means that as TensorFlow is updated in the future, we'll be able to take
advantage of improvements to the LSTM model without modifying our code.
'''

# Import relevant libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Model parameters
numberOfHiddenNeurons = 10

############################################
# Helper functions to load/split data, and plot final results
############################################

# Load data (for EEG data)
def load_series(filename, length=100, electrode=10):
    try:
        with open(filename) as csvfile:
            data = [] # Initialise variable to store data
            csvreader = csv.reader(csvfile)
            # Get random EEG index
            lengthEEG = 1301-length
            randomIndex = int(np.floor(np.random.rand(1)*lengthEEG))
            for idx, row in enumerate(csvreader):
                if idx>0:
                    if idx>randomIndex & idx<randomIndex+length:
                        data.append(float(row[electrode]))
            normalized_data = (data - np.mean(data)) / np.std(data)
            return normalized_data
    except IOError:
        return None

# # Load data (for book data)
# def load_series(filename, series_idx=1):
#     try:
#         with open(filename) as csvfile:
#             # Load data
#             csvreader = csv.reader(csvfile)
#             # Loop through the lines of the file and convert to a floating point number
#             data = [float(row[series_idx]) for row in csvreader if len(row) > 0]
#             # Pre-process the data by mean-centering and dividing by standard deviation
#             normalized_data = (data - np.mean(data)) / np.std(data)
#         return normalized_data
#     except IOError:
#         return None

# Split data
def split_data(data, percent_train=0.60):
    num_rows = len(data)
    train_data, test_data = [], []
    # Loop over data
    for idx, row in enumerate(data): # Note the use of enumerate() to get index and content
        if idx < num_rows * percent_train:
            train_data.append(row)
        else:
            test_data.append(row)
    return train_data, test_data

# Plot final results
def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    #plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


############################################
# SeriesPredictor class
############################################
class SeriesPredictor:

    # Constructor function
    def __init__(self, input_dim, seq_size, hidden_dim=numberOfHiddenNeurons):
        # Hyperparameters
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_dim = hidden_dim

        # Weight variables and input placeholders
        self.W_out = tf.Variable(tf.random_normal([hidden_dim, 1]), name='W_out')
        self.b_out = tf.Variable(tf.random_normal([1]), name='b_out')
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        # Cost optimizer
        self.cost = tf.reduce_mean(tf.square(self.model() - self.y)) # Squared error
        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        # Auxiliary ops
        self.saver = tf.train.Saver()

    # Define the RNN model
    def model(self):
        '''self.x: inputs of size [T, batch_size, input_size]
        self.W: matrix of fully-connected output layer weights
        self.b: vector of fully-connected output layer biases'''

        # Create a LSTM cell
        cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, reuse=tf.get_variable_scope().reuse)
        
       #  Run the cell on the input to obtain tensors for outputs (and states)
        num_examples = tf.shape(self.x)[0]
        outputs, states = tf.nn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        
        #  Compute the output layer as a fully connected linear function
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_examples, 1, 1])
        out = tf.matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        return out

    # Train the model (i.e. learn the LSTM weights given example input/output pairs)
    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.global_variables_initializer())
            max_patience = 3
            patience = max_patience
            min_test_err = float('inf')
            step = 0
            while patience > 0:
                _, train_err = sess.run([self.train_op, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                if step % 100 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: test_y})
                    print('step: {}\t\ttrain err: {}\t\ttest err: {}'.format(step, train_err, test_err))
                    if test_err < min_test_err:
                        min_test_err = test_err
                        patience = max_patience
                    else:
                        patience -= 1
                step += 1
            save_path = self.saver.save(sess, './modelData/model.ckpt')
            print('Model saved to {}'.format(save_path))

    # Test the model
    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, './modelData/model.ckpt')
        output = sess.run(self.model(), feed_dict={self.x: test_x})
        return output

############################################
# Run the RNN script
############################################

if __name__ == '__main__':

    # Common parameters
    seq_size = 5

    # Create SeriesPredictor instance
    predictor = SeriesPredictor(
        input_dim = 1, # The dimension of each element of the sequence is a scalar (1-dimensional)
        seq_size = seq_size, # Length of each sequence
        hidden_dim = numberOfHiddenNeurons) # Size of the RNN hidden dimension
    
    # Load the data and split into training and test data
    # data = load_series('international-airline-passengers.csv')
    data = load_series('eegData.csv')
    train_data, test_data = split_data(data)

    # # Function to create a list of sequences (length = seq_size)
    ''' This function loops through the time-series data and, with
    every iteration, takes a 5-long sequence from the current window.
    The defined window moves one step to the right on every iteration'''
    train_x, train_y, test_x, test_y = [], [], [], []
    for i in range(len(train_data) - seq_size - 1):
        train_x.append(np.expand_dims(train_data[i:i+seq_size], axis=1).tolist())
        train_y.append(train_data[i+1:i+seq_size+1])
    for i in range(len(test_data) - seq_size - 1):
        test_x.append(np.expand_dims(test_data[i:i+seq_size], axis=1).tolist())
        test_y.append(test_data[i+1:i+seq_size+1])
    # def createListOfSequences(sourceData): # I don't know why this doesn't work! (ValueError: setting an array element with a sequence.)
    #     x_data, y_data = [], []
    #     for i in range(len(train_data) - seq_size - 1):
    #         x_data.append(np.expand_dims(sourceData[i:i+seq_size], axis=1).tolist())
    #         y_data.append(sourceData[i+1:i+seq_size+1])
    #     return x_data, y_data
    # train_x, train_y = createListOfSequences(train_data)
    # test_x, test_y = createListOfSequences(test_data)

    # Train the model
    predictor.train(train_x, train_y, test_x, test_y)

    # Test the model
    with tf.Session() as sess:
        # Get and plot the predicted values (from the full data)
        '''In this test, we give the full time series data to the model'''
        predicted_vals = predictor.test(sess, test_x)[:,0]
        print('predicted_vals', np.shape(predicted_vals))
        # plot_results(train_data, predicted_vals, test_data, 'predictions.png')
        plot_results(train_data, predicted_vals, test_data, None)

        # Get and plot the predicted values (from just the last sequence)
        predicted_vals = []
        prev_seq = train_x[-1] # Get just the last test sequence (length=5)
        for i in range(20):
            next_seq = predictor.test(sess, [prev_seq])
            predicted_vals.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        # plot_results(train_data, predicted_vals, test_data, 'hallucinations.png')
        plot_results(train_data, predicted_vals, test_data, None)


# # Get random EEG index
# lengthEEG = len(list(csvreader))-dataLength
# randomIndex = int(np.floor(np.random.rand(1)*lengthEEG))