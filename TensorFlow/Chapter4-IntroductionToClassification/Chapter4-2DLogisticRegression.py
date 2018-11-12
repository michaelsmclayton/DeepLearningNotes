# Logistic regression in higher dimensions
''' With the fundamental principles of logistic regression previously applied
in the simple contex of one-dimensional data, we can now learn how use logistic
regression with multiple independent variables. The number of independent variables
corresponds to the number of dimensions. Consider for example, that crime events
are plotted on a 2D graph (with the x-dimension giving lattitude and the y dimension
giving longitude). The activities of two gangs can be shown in different colours,
and seem to center on two different parts of the city. If we then hear about a new
crime and its location, can we determine which gang was most likely responsible
for this crime?

    For this crime, we have two independent variables (x1 and x2; i.e. lattitude and
longitude). A simple way to model the mapping between the input x and output M(x) is
the following equation, where w is the parameters to be found using TensorFlow.

    M(x; w) = sigmoid(w2*x2 + w1*x1 + w0)
'''

# Import the usual libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Create fake data
x1_label1 = np.random.normal(3, 1, 1000) # Group 1
x2_label1 = np.random.normal(2, 1, 1000)
x1_label2 = np.random.normal(7, 1, 1000) # Group 2
x2_label2 = np.random.normal(6, 1, 1000)
x1s = np.append(x1_label1, x1_label2)
x2s = np.append(x2_label1, x2_label2)
ys = np.asarray([0.] * len(x1_label1) + [1.] * len(x1_label2))

# Define the hyper-parameters (note that these parameters are different from all previous examples)
learning_rate = 0.1
training_epochs = 2000

# Define placeholders, variables, model, and the training op:
X1 = tf.placeholder(tf.float32, shape=(None,), name="x1")
X2 = tf.placeholder(tf.float32, shape=(None,), name="x2")
Y = tf.placeholder(tf.float32, shape=(None,), name="y")

# Initialise weights
w = tf.Variable([0., 0., 0.], name="w", trainable=True)

# Define the model; M(x; w) = sigmoid(w2*x2 + w1*x1 + w0)
y_model = tf.sigmoid(-(w[2]*X2 + w[1]*X1 + w[0]))

# Define the cost function
cost = tf.reduce_mean(-tf.log(y_model * Y + (1 - y_model) * (1 - Y)))

# Set up the training op
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Train the model on the data in a session:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_err = 0
    for epoch in range(training_epochs):
        err, _ = sess.run([cost, train_op], {X1: x1s, X2: x2s, Y: ys})
        if epoch % 100 == 0:
            print(epoch, err)
        if abs(prev_err - err) < 0.0001:
            break
        prev_err = err
    w_val = sess.run(w, {X1: x1s, X2: x2s, Y: ys})

# Here's one hacky, but simple, way to figure out the decision boundary of the classifier:
x1_boundary, x2_boundary = [], []
with tf.Session() as sess:
    for x1_test in np.linspace(0, 10, 20):
        for x2_test in np.linspace(0, 10, 20):
            z = sess.run(tf.sigmoid(-x2_test*w_val[2] - x1_test*w_val[1] - w_val[0]))
            if abs(z - 0.5) < 0.05:
                x1_boundary.append(x1_test)
                x2_boundary.append(x2_test)

# Ok, enough code. Let's see some a pretty plot:
plt.scatter(x1_boundary, x2_boundary, c='b', marker='o', s=20)
plt.scatter(x1_label1, x2_label1, c='r', marker='x', s=20)
plt.scatter(x1_label2, x2_label2, c='g', marker='1', s=20)
plt.show()