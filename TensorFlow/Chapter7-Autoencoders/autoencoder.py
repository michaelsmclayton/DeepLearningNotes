import tensorflow as tf
import numpy as np

# Function to select a single batch for training
def get_batch(X, size):
    a = np.random.choice(len(X), size, replace=False)
    return X[a]

class Autoencoder:
    #####################################################
    # Initialise variables
    #####################################################
    ''' The constructor will set up all the TensorFlow variables, placeholders,
    optimizers, and operators. Anything that does not immediately need a session
    can go in the constructor'''
    def __init__(self, input_dim, hidden_dim, epochs=1000, batch_size=50, learning_rate=0.001):
        # Link epochs, batch_size, and learning_rate inputs with the object
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Define the input layer
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])

        # Define the weights, biases, and fundamental opererations for encoding/decoding
        ''' Because we are dealing with two sets of weights and biases (one for the
        encoding step and the other for the decoding), we can use TensorFlows name
        scopes to disambiguate a variables name. Now we can seamlessly save and restore
        this variable without worrying about name-collisions'''
        with tf.name_scope('encode'):
            weights = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            encoded = tf.nn.sigmoid(tf.matmul(x, weights) + biases)
        with tf.name_scope('decode'):
            weights = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='weights')
            biases = tf.Variable(tf.zeros([input_dim]), name='biases')
            decoded = tf.matmul(encoded, weights) + biases

        # 'Method variables'
        self.x = x
        self.encoded = encoded
        self.decoded = decoded

        # Define the reconstruction cost function
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded))))

        # Choose the optimiser, and define the training operations.
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        
        # Setup a saver to save model parameters as they are being learned
        self.saver = tf.train.Saver()

    #####################################################
    # Train on a dataset
    #####################################################
    ''' Here, we will define a class method called train() that will receive a dataset
    and learn parameters to minimize its loss.
    '''
    def train(self, data):
        # Start a TensorFlow session
        with tf.Session() as sess:
            # Initialise all variables
            sess.run(tf.global_variables_initializer())
            # Loop through number of cycles/epochs
            for i in range(self.epochs):
                # Loop through the data in batches
                for j in range(np.shape(data)[0] // self.batch_size):
                    # Get a new batch
                    batch_data = get_batch(data, self.batch_size)
                    # Run the optimizer on the randomly selected batch
                    l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: batch_data})
                if i % 10 == 0: # Once every 10 cycles
                    print('epoch {0}: loss = {1}'.format(i, l)) # Print the reconstruction error
                    self.saver.save(sess, './modelData/model.ckpt') # Save the learned parameters to a file
            # Save the final parameters once loop has finished
            self.saver.save(sess, './modelData/model.ckpt')
        
    #####################################################
    # Test on some new data
    #####################################################
    ''' Here, we will design a clas to evaluate the autoencoder on new data'''
    def test(self, data):
        with tf.Session() as sess:
            # Load the saved, learned parameters
            self.saver.restore(sess, './modelData/model.ckpt')
            # Reconstruct the input
            hidden, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: data})
        # Print outputs
        print('input', data)
        print('compressed', hidden)
        print('reconstructed', reconstructed)
        return reconstructed # Return deconstructed data

    def get_params(self):
        with tf.Session() as sess:
            self.saver.restore(sess, './modelData/model.ckpt')
            weights, biases = sess.run([self.weights1, self.biases1])
        return weights, biases

    def classify(self, data, labels):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, './modelData/model.ckpt')
            hidden, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: data})
            reconstructed = reconstructed[0]
            # loss = sess.run(self.all_loss, feed_dict={self.x: data})
            print('data', np.shape(data))
            print('reconstructed', np.shape(reconstructed))
            loss = np.sqrt(np.mean(np.square(data - reconstructed), axis=1))
            print('loss', np.shape(loss))
            horse_indices = np.where(labels == 7)[0]
            not_horse_indices = np.where(labels != 7)[0]
            horse_loss = np.mean(loss[horse_indices])
            not_horse_loss = np.mean(loss[not_horse_indices])
            print('horse', horse_loss)
            print('not horse', not_horse_loss)
            return hidden

    def decode(self, encoding):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, './modelData/model.ckpt')
            reconstructed = sess.run(self.decoded, feed_dict={self.encoded: encoding})
        img = np.reshape(reconstructed, (32, 32))
        return img