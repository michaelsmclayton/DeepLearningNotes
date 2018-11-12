# Using logistic regression for classification
''' In contrast to linear regression (which is sensitive to outliers), logistic regression
provides us with an analytic function with theoretical guarantees on accuracy and performance.
It is just like linear regression, except we use a different cost function and slightly transform
the model response function.
    In linear regression, a line with non-zero slope may range from negative infinity to infinity.
However, if the only sensible results for classification are 0 or 1, then it would be intuitive to
instead fit a function with that property. Fortunately, the sigmoid function works well for this
because it converges to 0 or 1 very quickly. When x is 0, the sigmoid function results in 0.5.
As x increases to positive infinity, the function converges to 1. And as x decrease to negative
infinity, the function converges to 0. It turns out that the best-fit parameters of this function
imply a linear separation between the two classes. This separating line is also called a linear
decision boundary.
'''

# Defining the cost function for logistic regression
''' Although we could use the same cost function as linear regression, it would not be as fast, or
guarantee an optimal solution. To reason for this is the sigmoid function, which causes the cost
function to have many "bumps". As our aim is always to minimise the cost function, bumps in that
function can cause us to settle incorrectly at local minima. Instead, what we want is a CONVEX
cost function, which has no bumps and therefore which will always allow us to find the global
minimum. The new cost function that allows for optimal training with logistic regression is
as follows:

    cost(y, h) = -log(h);   if y = 1
                 -log(1-h); if y = 0

    This can be condenses into a single equation as follows:

    cost(y,h) = -y * log(h) - (1-y) * log(1-h)

    We use -log(h) when we want our output value to be 1 becuase this function is high at 0 and
low at 1. As it is a cost function, it will therefore penalise values with outputs of ~0, while
promoting values with outputs of ~1. In contrast, use use -log(1-h) when we want our output value
to be 0 becuase this function is low at 0 and high at 1. Conversely, this will mean that the cost
function will penalise values with outputs of ~1, while promoting values with outputs of ~0 (see
'Logistic regression cost functions.png').

    The benefit of using a sigmoud curve is that it is limited between 0 and 1, and rapidly
accelerates towards these extreme values when this input is less or greater than 0.5. This means,
in contrast to linear regression, that sigmoid curve fits are less sensitive to outliers.
'''

# Import the usual libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Set up some data to work with
x1 = np.random.normal(-4, 2, 1000)
x2 = np.random.normal(4, 2, 1000)
xs = np.append(x1, x2)
ys = np.asarray([0.] * len(x1) + [1.] * len(x2))

# Define the hyper-parameters
learning_rate = 0.01
training_epochs = 1000

# Define placeholders
X = tf.placeholder(tf.float32, shape=(None,), name="x")
Y = tf.placeholder(tf.float32, shape=(None,), name="y")

# Initialise weights
w = tf.Variable([0., 0.], name="parameter", trainable=True)

# Define the model using TensorFlows sigmoid function
y_model = tf.sigmoid(w[1] * X + w[0])

# Given a model, define the cost function (as defined in the notes above)
cost = tf.reduce_mean(-Y * tf.log(y_model) - (1 - Y) * tf.log(1 - y_model))

# Set up the training op
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Train the logistic model on the data:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_err = 0
    for epoch in range(training_epochs):
        err, _ = sess.run([cost, train_op], {X: xs, Y: ys})
        if epoch % 100 == 0:
            print(epoch, err)
        if abs(prev_err - err) < 0.0001:
            break
        prev_err = err
    w_val = sess.run(w, {X: xs, Y: ys})

# Now let's see how well our logistic function matched the training data points:
all_xs = np.linspace(-10, 10, 100)
with tf.Session() as sess:
    predicted_vals = sess.run(tf.sigmoid(all_xs * w_val[1] + w_val[0]))
plt.plot(all_xs, predicted_vals)
plt.scatter(xs, ys)
plt.show()
