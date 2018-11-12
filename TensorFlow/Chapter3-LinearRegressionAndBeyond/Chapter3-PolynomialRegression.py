''' From our script for linear regression in TF, the rest of the topics in regression are
conveniently just minor modifications. Making further improvements to the simple regression
model is simply a matter of enhancing the model with the right medley of variance and bias
as we discussed earlier. For example. the linear regression model we have designed so far is
burdened with a strong bias, meaning it only expresses a limited set of functions (i.e. linear
functions. Here, we will y_train a more flexible model. Specifically, we will make a POLYNOMIAL
MODEL. You will notice how only the TensorFlow graph needs to be rewired, while everything
else (such as preprocessing, training, evaluation) stays the same.
'''

#########################################################################
#              Running linear regression using TensorFlow
#########################################################################
''' When data points appear to form smooth curves rather than straight lines,
we need to change our regression model from a straight line to something else.
A polynomial is a generalization of a linear function. The nth degree polynomial
looks like: f(x) = w(n)*x(n) + ... + w(1)*x(1) + w(0)
'''

# Import the relevant libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define learning hyperparameters
learning_rate = 0.01
training_epochs = 40

# Generate fake data
x_train = np.linspace(-1, 1, 101) # input data (i.e. x; linearly spaced data between -1 and 1)
y_train = 0 # output data (i.e. y; based on a degree n polynomial)
num_coeffs = 6
y_train_coeffs = np.linspace(1, num_coeffs, num_coeffs) # [1, 2, 3, 4, 5, 6]
for i in range(num_coeffs):
    y_train += y_train_coeffs[i] * np.power(x_train, i)

# Add some noise
y_train += np.random.randn(*x_train.shape) * 1.5

# Define the nodes to hold values for input/output pairs
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Initialise the parameter weights vector with all zeros
w = tf.Variable([0.] * num_coeffs, name="parameters")

# Define our polynomial model
def model(X, w):
    terms = []
    for i in range(num_coeffs): # Looping through the number of coefficient
        term = tf.multiply(w[i], tf.pow(X, i))
        terms.append(term)
    return tf.add_n(terms) # Return the addition of all input tensors element-wise
y_model = model(X, w)

# Define the cost function just as before
cost = tf.reduce_sum(tf.square(Y-y_model))

# Define the operation that will be called on each iteration of the learning algorithm
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Set up the session and run the learning algorithm just as before
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Iteratively find the optimal model parameters
for epoch in range(training_epochs): # Loop through the dataset multiple times
    for (x, y) in zip(x_train, y_train): # Loop through each item in the dataset
        sess.run(train_op, feed_dict={X: x, Y: y})  # Update the model parameter(s) to y_train to minimize the cost function

# Obtain the final parameter value
w_val = sess.run(w)
# print(w_val) # [ 1.10158885  2.36433625  3.30378437  4.43473864  3.75751448  4.60356045]

# Close the session when done
sess.close()

# Plot the result
plt.scatter(x_train, y_train)
y_train2 = 0
for i in range(num_coeffs):
    y_train2 += w_val[i] * np.power(x_train, i)
plt.plot(x_train, y_train2, 'r')
plt.show()
