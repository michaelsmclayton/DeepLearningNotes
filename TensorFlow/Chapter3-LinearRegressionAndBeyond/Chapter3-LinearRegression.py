''' Regression is a study of how to best fit a curve to summarize data. The inputs
to a regression function can be continuous or discrete. However, the output must
always be continuous. Discrete-valued outputs are handled better by classification
techniques.

    To measure the success of a learning algorithm like regression, you need to
look at variance and bias:

    VARIANCE is how sensitive a prediction is to what training set was used.
    Ideally, how we choose the training set should not matter, meaning a lower
    variance is desired.

    BIAS is the strength/accuracy of assumptions made about the training dataset. Maing
    too many assumptions might make it hard to generalise.

    Models can OVERFIT data, in which, due to excessive flexible in the model parameters,
the model perfectly predicts the training data, but performs poorly on test data.
Alternatively, models can UNDERFIT data, in which a not-so-flexible model may
generalize better to unseen testing data, but would score relatively low on the training
data. A too flexible model (i.e. overfit) has high variance and low bias, whereas a too
strict model (underfit) has low variance and high bias. Ideally we would like a model with
both low variance error and low bias error. That way, it both generalizes to unseen data
and captures the regularities of the data. Concretely, the variance of a model is a measure
of how badly the responses fluctuate, and the bias is a measure of how badly the response is
offset from the ground-truth.
'''

#########################################################################
#              Running linear regression using TensorFlow
#########################################################################
''' To fit a model, you need to provide TensorFlow with a score for each
candidate parameter it tries. This score assignment is commonly called a
cost function. The higher the cost, the worse the model parameter will be.
After we define the situation as a cost minimization problem, TensorFlow
takes care of the inner workings and tries to update the parameters in an
efficient way to eventually reach the best possible value. Each step of
looping through all your data to update the parameters is called an EPOCH.
In the example below, we define cost as sum of squares error (i.e. sum of
squared differences between the actual and predicted values).
'''

# Import dependencies
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np # Import NumPy to help generate initial raw data
import matplotlib.pyplot as plt # Use matplotlib to visualize data

# Define learning hyperparameters
learning_rate = 0.01
training_epochs = 100

# Generate fake data
x_train = np.linspace(-1, 1, 101) # Creates input values of 101 evenly spaced numbers -1 and 1
y_train = 2 * x_train # Have the output values be linearly proportional to the input

# Add some noise to the output data
y_train += np.random.randn(*x_train.shape) * 0.33 

# Define the nodes to hold values for input/output pairs
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Set up the weights variable
w = tf.Variable(0.0, name="weights")

# Define the model as y = w*x
def model(X, w):
    return tf.multiply(X, w)
y_model = model(X, w)

# Define the cost function
cost = tf.square(Y-y_model)

# Define the operation that will be called on each iteration of the learning algorithm
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Set up a session and initialize all variables
sess = tf.Session()
init = tf.global_variables_initializer() # initialize all trainable variables in one go
sess.run(init)

# Iteratively find the optimal model parameters
for epoch in range(training_epochs): # Loop through the dataset multiple times
    for (x, y) in zip(x_train, y_train): # Loop through each item in the dataset
        sess.run(train_op, feed_dict={X: x, Y: y}) # Update the model parameter(s) to try to minimize the cost function

# Obtain the final parameter value
w_val = sess.run(w)

# Close the session
sess.close()

# Plot data with best fitting line
plt.scatter(x_train, y_train) # Call matplotlib function to generate a scatter plot of the data
y_learned = x_train*w_val
plt.plot(x_train, y_learned, 'r')
plt.show() # Show plot