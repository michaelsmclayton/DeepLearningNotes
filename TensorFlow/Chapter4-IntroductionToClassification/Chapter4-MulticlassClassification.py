'''The previous script on 2D classification dealt with multidimensional input. In this script, we
will focus on multivariate output (i.e. multi-class classification). For example, instead of binary
labels on the data, what if we have 3, or 4, or 100 classes? To handle more than two labels, we may
reuse logistic regression in a clever way (using a one-versus-all or one-versus-one approach) or
develop a new approach (softmax regression). The logistic regression approaches require a decent
amount of ad-hoc engineering, so we will focus our efforts on softmax regression.
'''

# One versus all, and one versus one approaches
''' One way to perform multivariate classification is to have a detector for each class. For
example, if there are three labels, we have three classifiers available to use: f1, f2, and f3.
To test on new data, we run each of the classifiers to see which one produced the most confident
response. Intuitively, we label the new point by the label of the classifier that responded most
confidently. In contrast, we could train a classifier to distinguish between pairs of labels.
For example, if there are three labels, then that is just three unique pairs. But for k number of
labels, that is k(k-1)/2 pairs of labels. On new data, we run all the classifiers and choose the
class with the most wins.
'''

# Softmax regresssion
''' Softmax is named after the traditional max function, which takes a vector and returns the max
value; however, softmax is not exactly the max function because it has the added benefit of being
continuous and differentiable. As a result, it has the helpful properties for stochastic gradient
descent to work efficiently. In this type of multiclass classification setup, each class has a
confidence (or probability) score for each input vector. The softmax step simply picks the highest
scoring output.
'''

# Import the usual libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#######################################################
# Setting up training and test data for multiclass classification
#######################################################

# Generate training data
x1_label0 = np.random.normal(1, 1, (100, 1)) # Class 1
x2_label0 = np.random.normal(1, 1, (100, 1))
x1_label1 = np.random.normal(5, 1, (100, 1)) # Class 2
x2_label1 = np.random.normal(4, 1, (100, 1))
x1_label2 = np.random.normal(8, 1, (100, 1)) # Class 3
x2_label2 = np.random.normal(0, 1, (100, 1))
xs_label0 = np.hstack((x1_label0, x2_label0)) # Class 1
xs_label1 = np.hstack((x1_label1, x2_label1)) # Class 2
xs_label2 = np.hstack((x1_label2, x2_label2)) # Class 3
xs = np.vstack((xs_label0, xs_label1, xs_label2)) # Combine the data altogether

# Define the labels and shuffle the data:
'''The labels must be represented as a vector where only one element is 1 and the rest are 0s. This representation is called one-hot encoding'''
labels = np.matrix([[1., 0., 0.]] * len(x1_label0) + [[0., 1., 0.]] * len(x1_label1) + [[0., 0., 1.]] * len(x1_label2))
arr = np.arange(xs.shape[0])
np.random.shuffle(arr)
xs = xs[arr, :]
labels = labels[arr, :]

# Generate test data
test_x1_label0 = np.random.normal(1, 1, (10, 1))
test_x2_label0 = np.random.normal(1, 1, (10, 1))
test_x1_label1 = np.random.normal(5, 1, (10, 1))
test_x2_label1 = np.random.normal(4, 1, (10, 1))
test_x1_label2 = np.random.normal(8, 1, (10, 1))
test_x2_label2 = np.random.normal(0, 1, (10, 1))
test_xs_label0 = np.hstack((test_x1_label0, test_x2_label0))
test_xs_label1 = np.hstack((test_x1_label1, test_x2_label1))
test_xs_label2 = np.hstack((test_x1_label2, test_x2_label2))
test_xs = np.vstack((test_xs_label0, test_xs_label1, test_xs_label2))
test_labels = np.matrix([[1., 0., 0.]] * 10 + [[0., 1., 0.]] * 10 + [[0., 0., 1.]] * 10)

train_size, num_features = xs.shape

#######################################################
# Perform softmax regression
#######################################################

# Define the hyper-parameters
learning_rate = 0.01
training_epochs = 1000
num_labels = 3
batch_size = 100 # Batch learning is used here

# Define placeholders
X = tf.placeholder("float", shape=[None, num_features])
Y = tf.placeholder("float", shape=[None, num_labels])

# Define the model parameters
W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))

# Define model
y_model = tf.nn.softmax(tf.matmul(X, W) + b)

# Define cost function
cost = -tf.reduce_sum(Y * tf.log(y_model))

# Set up the training op
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Define an op to measure success rate
correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Train the softmax classification model:
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for step in range(training_epochs * train_size // batch_size):
        offset = (step * batch_size) % train_size
        batch_xs = xs[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size)]
        err, _ = sess.run([cost, train_op], feed_dict={X: batch_xs, Y: batch_labels})
        if step % 100 == 0:
            print (step, err)
            
    W_val = sess.run(W)
    print('w', W_val)
    b_val = sess.run(b)
    print('b', b_val)
    print("accuracy", accuracy.eval(feed_dict={X: test_xs, Y: test_labels}))

    # # Get prediction
    # feed_dict = {X: xs[:,1]}
    # classification = sess.run(y_model, feed_dict)
    # print classification

####################################################
# Use the model to make class predictions
####################################################
with tf.Session() as sess: # Start a session
    tf.global_variables_initializer().run()

    # Create input data for prediction
    xPosition = 5
    yPosition = 5
    inputData = np.array([[xPosition, yPosition]], dtype=np.float32)
    
    # Get model prediction
    modelPredictions = sess.run(y_model, {X: inputData, W: W_val, b: b_val})
    print(modelPredictions)
    strongestPrediction = np.argmax(modelPredictions)

    # Print predictions 
    if strongestPrediction == 0:
        print('Group 1 (Red)')
    elif strongestPrediction == 1:
        print('Group 2 (Green)')
    else:
        print('Group 3 (Blue)') 

# Plot data (with prediction data)
plt.scatter(x1_label0, x2_label0, c='r', marker='o', s=60)
plt.scatter(x1_label1, x2_label1, c='g', marker='x', s=60)
plt.scatter(x1_label2, x2_label2, c='b', marker='_', s=60)
plt.scatter(inputData[0,0], inputData[0,1], c='b', marker='o', s=100) # Show prediction input data
plt.show()