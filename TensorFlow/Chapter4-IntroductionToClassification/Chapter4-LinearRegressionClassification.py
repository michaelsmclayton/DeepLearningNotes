# Using linear regression for classification
''' One of the simplest ways to implement a classifier is to tweak a linear
regression algorithm. Remember that a regression function takes continuous
real numbers as input and produces continuous real numbers as output. As
classification is all about discrete outputs, we will need to force the
regression model to produce a two-valued (in other words, binary) output. We
can do this by setting values above some threshold to a number (such as 1) and
values below that threshold to a different number (such as 0). This could be
refered to as a threshold approach.

    Linear regression can work well when you're training nicely fits a straight
line. However, the linear regression approach fails miserably if we train on
more extreme data (i.e. with outliers). This is demonstrated in the script below,
where the addition of just one outlier (x_label0 = np.append(np.random.normal(5, 1, 9), 20))
significantly reduces the accuracy of the classifier.
'''

# Import the usual libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

################################################
# Set up linear regression (for classification)
################################################

# Create fake data for two classes (with 1-dimensional values).
x_label0 = np.random.normal(5, 1, 10) # (mean, standard deviation, n)
# x_label0 = np.append(np.random.normal(5, 1, 9), 20) # Add outlier to break linear regression accuracy
x_label1 = np.random.normal(2, 1, 10) # 10 instances of each label
xs = np.append(x_label0, x_label1)
labels = [0.] * len(x_label0) + [1.] * len(x_label1) # Initialize the corresponding labels; Numbers close to 5 will be labelled [0], and numbers close to 2 will be labelled [1]

# Define the hyper-parameters
learning_rate = 0.001
training_epochs = 1000

# Define placeholders
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Initialise weights
w = tf.Variable([0., 0.], name="parameters")

# Define the linear model (y = w1 * x + w0)
def model(X, w):
    return tf.add(tf.multiply(w[1], tf.pow(X, 1)), # w1 * x
                  tf.multiply(w[0], tf.pow(X, 0))) # + w0
y_model = model(X, w)

# Given a model, define the cost function
cost = tf.reduce_sum(tf.square(Y-y_model)) # Note that Y = labels

################################################
# Executing the graph
################################################
''' The train_op updates the models parameters to better and better guesses. We run the train_op
multiple times in a loop since each step iteratively improves the parameter estimate. 
'''

# Set up the training op
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# To measure success, we can count the number of correct predictions and compute a success rate
correct_prediction = tf.equal(Y, tf.to_float(tf.greater(y_model, 0.5))) # When the models response is greater than 0.5, it should be a positive label, and vice versa
accuracy = tf.reduce_mean(tf.to_float(correct_prediction)) # Compute the percent of success

# Prepare the session:
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run the training op multiple times on the input data:
for epoch in range(training_epochs):
    sess.run(train_op, feed_dict={X: xs, Y: labels})
    current_cost = sess.run(cost, feed_dict={X: xs, Y: labels})
    if epoch % 100 == 0:
        print(epoch, current_cost)
    # 0 8.63226
    # 100 3.23953
    # 200 2.14632
    # 300 1.90881
    # 400 1.8572
    # 500 1.84599
    # 600 1.84356
    # 700 1.84303
    # 800 1.84291
    # 900 1.84289

# Show some final metrics/results:
w_val = sess.run(w)
print('learned parameters', w_val)
print('accuracy', sess.run(accuracy, feed_dict={X: xs, Y: labels}))
# ('learned parameters', array([ 1.3983544, -0.2374023], dtype=float32))
# ('accuracy', 0.95)

# Close the session
sess.close()

# Plot the learned function
all_xs = np.linspace(0, 10, 100)
plt.plot(all_xs, all_xs*w_val[1] + w_val[0])
plt.scatter(xs, labels)
plt.show()
