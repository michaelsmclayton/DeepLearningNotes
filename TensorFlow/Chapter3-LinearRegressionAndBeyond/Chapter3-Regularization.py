''' The previous script showed how to fit polynomial models using TensorFlow.
However, just because these polynomial models are very flexible, does not mean
that we should always favour them. Use of such complex models can easily lead
to overfitting. To avoid this problem, we can use REGULARISATION, which  is a
technique to structure the model parameters in a form we prefer. More specifically,
we can use regularisation to encourage the learning algorithm to produce a smaller
coefficient vector (lets call it w). We do this by adding a penalty to the loss
term, meaning that parameters vectors with a large number of larger weights will
be disfavoured over a model with a smaller number of small weights. We can control
how significantly we want to weigh the penalty term by multiply the penalty by a
constant non-negative number, lambda. When lambda is 0, regularisation is not in
play. When it is large, parameters with larger norms will be heavily penalized.
Simply put, regularization reduces some of the flexibility of the otherwise easily
tangled model.
'''

# Import the relevant libraries and initialize the hyper-parameters
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Seed the random number generator
np.random.seed(100)

# Define learning hyperparameters
learning_rate = 0.001
training_epochs = 1000
regularisationLambda = 0.

# Create a helper method to split the dataset
def split_dataset(x_dataset, y_dataset, ratio): # Take the input and output dataset as well as the desired split ratio
    arr = np.arange(x_dataset.size) # Create an array the sizs of x_dataset
    np.random.shuffle(arr) # Shuffle this array
    num_train = int(ratio * x_dataset.size) # Calculate the number of training examples
    x_train = x_dataset[arr[0:num_train]] # Use the shuffled list to create the training dataset
    y_train = y_dataset[arr[0:num_train]]
    x_test = x_dataset[arr[num_train:x_dataset.size]] # Use the remaining data to create the test dataset
    y_test = y_dataset[arr[num_train:x_dataset.size]]
    return x_train, x_test, y_train, y_test

# Create a fake data
x_dataset = np.linspace(-1, 1, 100) # input data (i.e. x, linearly spaced data between -1 and 1)
y_dataset = 0 # output data (i.e. y, based on a degree n polynomial)
num_coeffs = 9
y_dataset_params = [0.] * num_coeffs
y_dataset_params[2] = 1
for i in range(num_coeffs):
    y_dataset += y_dataset_params[i] * np.power(x_dataset, i)

# Add some noise
y_dataset += np.random.randn(*x_dataset.shape) * 0.3

# Split the dataset into 70% training and testing 30%
(x_train, x_test, y_train, y_test) = split_dataset(x_dataset, y_dataset, 0.7)

# Set up the input/output placeholders
X = tf.placeholder("float")
Y = tf.placeholder("float")
#
# Initialise the parameter weights vector with all zeros
w = tf.Variable([0.] * num_coeffs, name="parameters")

# Define our model
def model(X, w):
    terms = []
    for i in range(num_coeffs):
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms)
y_model = model(X, w)

# Define the regularized cost function
squareError = tf.reduce_sum(tf.square(Y-y_model))
regularisation = tf.multiply(regularisationLambda, tf.reduce_sum(tf.square(w))) # looks like L2 norm
cost = tf.div(tf.add(squareError, regularisation), 2*x_train.size)

# Define the operation that will be called on each iteration of the learning algorithm
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Set up the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Try out various regularization parameters
''' To figure out which value of the regularization parameter lambda performs best,
we can simply loop through lambda values (e.g. from 0 to 100), and see when the
cost is minimised.'''
for regularisationLambda in np.linspace(0,1,100):
    for epoch in range(training_epochs):
        sess.run(train_op, feed_dict={X: x_train, Y: y_train})
    final_cost = sess.run(cost, feed_dict={X: x_test, Y:y_test})
    print('reg lambda', regularisationLambda)
    print('final cost', final_cost)

'''What you will find is that, when lambda is greater, cost is reduced'''
# ('reg lambda', 0.010101010101010102)
# ('final cost', 0.028902497)
# ('final cost', 0.023091594)
# ('reg lambda', 0.08080808080808081)