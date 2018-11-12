''' Machine learning is an eternal struggle of designing a model that is expressive
enough to represent the data, yet not so flexible that it overfits and memorizes the
patterns. A quick and dirty heuristic you can use to compare the flexibility of two
machine learning models is to count the number of parameters to be learned. For example,
in a fully connected neural network that takes in a 256x256 image and maps it to a layer
of 10 output neurons, you will have 256*256*10 = 655360 parameters! Convolutional neural
networks are a clever way to reduce the number of parameters in this kind of network.
Instead of dealing with a fully connected network, the CNN approach ends up reusing the
same parameter multiple times.

    The big idea behind convolutional neural networks is that a local understanding of
an image is good enough. Instead of a fully connected network of weights from each pixel,
a CNN has just enough weights to look only at a small patch of the image. Consider a
256x256 image. Instead of your TensorFlow code processing the whole image at the same
time, it can efficiently scan it chunk-by-chunk, with each chunk being say a 5x5 window.
The 5x5 window slides along the image (usually left-to-right and top-to-bottom). How 
"quickly" it slides is called its STRIDE-LENGTH. For example, a stride-length of 2 means
the sliding window jumps around by 2 pixels.

    A typical CNN has multiple convolution layers. Each convolutional layer typically
generates many alternate convolutions, so the weight matrix is actually a tensor of
5x5xn, where n is the number of convolutions. As an example, let's say an image goes
through a convolution layer on a weight matrix of 5x5x64. That means it generates 64
convolutions by sliding 64 5x5 windows. Therefore, this model has 5*5*64 (=1,600)
parameters, which is remarkably fewer parameters than a fully connected network,
256*256 (=65,536). This is the beauty of CNNs: the number of parameters is independent
of the size of the original image. We can run the same CNN on a 300x300 image, and
the number of parameters will not change in the convolution layer!
'''

# Tips for how to improve a CNN
''' AUGMENT DATA: From a single image, you can easily generate new training images. As a
start, just flip an image horizontally or vertically and you can quadruple your dataset size.
You may also adjust the brightness of the image or the hue to ensure the neural network
generalizes to other fluctuations. Lastly, you may even want to add random noise to the
image to make the classifier robot to small occlusions. Scaling the image up or down can
also be helpful; having exactly the same size items in your training images will almost
guarantee you overfitting!

    EARLY STOPPING: Keep track of the training and testing error while you train the neural
network. At first, both errors should slowly dwindle down, because the network is learning.
But sometimes, the test error goes back up. This is a signal that the neural network has
started overfitting on the training data, and is unable to generalize to previously unseen
input. You should stop the training the moment you witness this phenomenon.

    REGULARISE WEIGHTS: Another way to combat overfitting is by adding a regularization term
to the cost function. We've already seen regularization in previous chapters, and the same
concepts apply here.

    DROPOUT: TensorFlow comes with a handy function tf.nn.dropout, which can be applied to
any layer of the network to reduce overfitting. It turns off a randomly selected number of
neurons in that layer during training so that the network must be redundant and robust to
inferring output.

    DEPTH: A deeper architecture means adding more hidden layers to the neural network. If you
have enough training data, it's been shown that adding more hidden layers improves performance.
'''

# Import the usual dependencies
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cPickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
random.seed(1)

######################################################################################
#                                 PREPARING THE IMAGES
######################################################################################

# Define helper function to load data
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

##################################################
# Clean the data
##################################################
''' Neural networks are already prone to overfitting, so it's essential that you do
as much as you can to minimize that error. For that reason, always remember to clean
the data before processing it. In the function below, we clean the data by following
these steps:

    1. If you have an image in color, try converting it to grayscale instead to lower
    the dimensionality of the input data, and consequently lower the number of parameters.

    2. Consider center-cropping the image because maybe the edges of an image provide
    no useful information.

    3. Don't forget to normalize your input by subtracting the mean and dividing by the
    standard deviation of each data sample so that the gradients during back-propagation
    don't change too dramatically.
'''
def clean(data):
     # Reorganize the data so it is a 32x32 matrix with 3 channels
    imgs = data.reshape(data.shape[0], 3, 32, 32)

    # Grayscale the image by averaging the color intensities
    grayscale_imgs = imgs.mean(1)

    # Center-crop the images (i.e. cropping the 32x32 images to a 24x24 images)
    cropped_imgs = grayscale_imgs[:, 4:28, 4:28]
    img_data = cropped_imgs.reshape(data.shape[0], -1)
    img_size = np.shape(img_data)[1]

    # Normalise the pixel values by subtracting the mean and dividing by standard deviation
    means = np.mean(img_data, axis=1)
    meansT = means.reshape(len(means), 1)
    stds = np.std(img_data, axis=1)
    stdsT = stds.reshape(len(stds), 1)
    adj_stds = np.maximum(stdsT, 1.0 / np.sqrt(img_size))
    normalized = (img_data - meansT) / adj_stds

    # Return cleaned data
    cleanedData = normalized
    return cleanedData

##################################################
# Load the CIFAR-10 images (and call clean() function)
##################################################
''' There seems to be no distinction here between training, evaluation, and test data sets?
'''
def read_data(directory):
    # Get the images labels
    names = unpickle('{}/batches.meta'.format(directory))['label_names']
    print('names', names) # 'names', ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Load the CIFAR-10 image data
    data, labels = [], []
    for i in range(1, 6):
        filename = '{}/data_batch_{}'.format(directory, i)
        batch_data = unpickle(filename)
        if len(data) > 0:
            data = np.vstack((data, batch_data['data']))
            labels = np.hstack((labels, batch_data['labels']))
        else: # Add the batch data only if 'data' is empty
            data = batch_data['data']
            labels = batch_data['labels']
    print(np.shape(data), np.shape(labels)) # Print shape of data

    # Clean data
    data = clean(data)
    data = data.astype(np.float32)

    # Return list of possible labels, cleaned data, and labels for that data
    return names, data, labels

##################################################
# Display a random selection of CIFAR-10 images
##################################################
def show_some_examples(names, data, labels):
    plt.figure()
    rows, cols = 4, 4
    random_idxs = random.sample(range(len(data)), rows * cols)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        j = random_idxs[i]
        plt.title(names[labels[j]])
        img = np.reshape(data[j, :], (24, 24))
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('cifar_examples.png') # Save to .png

# Load and display data!
names, data, labels = read_data('../cifar-10-batches-py')
print('Label size = ', labels.shape)
# show_some_examples(names, data, labels)

######################################################################################
#                                  GENERATE FILTERS
######################################################################################

# Generate and visualize random filters (i.e. patches)
''' Here, we will randomly initialize 32 filters. We will do so by defining a variable called
W of size 5x5x1x32. The first two dimensions correspond to the filter size. The last dimension
corresponds to the 32 different convolutions. The "1" in the variable's size corresponds to the
input dimension, because the conv2d function is capable of convolving images of multiple channels.
(In our example, we only care about grayscale images, so number of input channels is 1)'''
W = tf.Variable(tf.random_normal([5, 5, 1, 32]))

# Function to display the filter weights
def show_weights(W, filename=None):
    plt.figure()
    rows, cols = 4, 8 # Define enough rows and columns to show the 32 figures 
    for i in range(np.shape(W)[3]):
        img = W[:, :, 0, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none') # Visualize each filter matrix
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


######################################################################################
#                               CONVOLVE USING FILTERS
######################################################################################
''' With the intial filters now generated, we can now use TensorFlow's convolve function
on these filters. The code below visualize the convolution outputs'''
def show_conv_results(data, filename=None):
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(data)[3]):
        img = data[0, :, :, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# Start with a random image (and visualise it)
raw_data = data[4, :]
raw_img = np.reshape(raw_data, (24, 24))
# plt.figure()
# plt.imshow(raw_img, cmap='Greys_r')
# plt.show()

# Define the input tensor for the 24x24 image
x = tf.reshape(raw_data, shape=[-1, 24, 24, 1])

# Intialise the biases with random numbers
b = tf.Variable(tf.random_normal([32]))

# Define the convolution steps
''' By adding a bias term and an activation function such as relu, the convolution layer of
the network behaves nonlinearly, which improves its expressiveness'''
conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # Convole image x with weights W (getting 32 resulting images)
conv_with_b = tf.nn.bias_add(conv, b) # Add bias terms
conv_out = tf.nn.relu(conv_with_b) # Use rectified linear function to get non-linear output

######################################################################################
#                                    MAX-POOLING
######################################################################################
''' After a convolution layer extracts useful features, it's usually a good idea to reduce
the size of the convolved outputs. Rescaling or subsampling a convolved output helps reduce
the number of parameters, which in turn can help to not overfit the data. This is the main
idea behind MAX-POOLING, which sweeps a window across an image and picks the pixel with the
maximum value. Depending on the stride-length, the resulting image is a fraction of the size
of the original (as only one pixel is kept at each stride). This is useful because it lessens
the dimensionality of the data, consequently lowering the number of parameters in future steps.
'''
k = 2
maxpool = tf.nn.max_pool(conv_out,
                         ksize=[1, k, k, 1],
                         strides=[1, k, k, 1],
                         padding='SAME')


######################################################################################
#                                DISPLAY THE RESULTS
######################################################################################

# # Display the weights
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     # Displauy the intial weights
#     W_val = sess.run(W)
#     print('weights:')
#     show_weights(W_val)

#     # Display the convolution results
#     conv_val = sess.run(conv)
#     print('convolution results:')
#     print(np.shape(conv_val)) # (1, 24, 24, 32)
#     show_conv_results(conv_val)
    
#     # Display convolution output
#     conv_out_val = sess.run(conv_out)
#     print('convolution with bias and relu:')
#     print(np.shape(conv_out_val)) # (1, 24, 24, 32)
#     show_conv_results(conv_out_val)

#     # Display max-pooling output
#     maxpool_val = sess.run(maxpool)
#     print('maxpool after all the convolutions:')
#     print(np.shape(maxpool_val)) # (1, 12, 12, 32)
#     show_conv_results(maxpool_val)
#     ''' After running max-pool, the convolved outputs are halved in size, making
#     the algorithm computationally faster without losing too much information.
#     '''

######################################################################################
#              IMPLEMENTING A CONVOLUTIONAL NEURAL NETWORK IN TENSORFLOW
######################################################################################
''' A convolutional neural network has multiple layers of convolutions and max-pooling.
The convolution layer offers different perspectives on the image, while the maxpooling
layer simplifies the computations by lowering the dimensionality without losing too much
information.
'''

# Learning hyper-parameter
learning_rate = 0.001


#########################################################
# Set up CNN weights and biases (for each layer)
#########################################################

# Define the input and output placeholders
x = tf.placeholder(tf.float32, [None, 24 * 24], name='input')
y = tf.placeholder(tf.float32, [None, len(names)], name='prediction')

# Apply 64 convolutions of window-size 5x5
W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]), name='W1')
b1 = tf.Variable(tf.random_normal([64]), name='b1')

# Then apply 64 more convolutions of window-size 5x5
W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]), name='W2')
b2 = tf.Variable(tf.random_normal([64]), name='b2')

# Then we introduce a fully-connected layer
# Does the convolution make 64 images of size (6*6)?
W3 = tf.Variable(tf.random_normal([6*6*64, 1024]), name='W3')
b3 = tf.Variable(tf.random_normal([1024]), name='b3')

# Lastly, define the variables for a fully-connected linear layer
W_out = tf.Variable(tf.random_normal([1024, len(names)]), name='W_out')
b_out = tf.Variable(tf.random_normal([len(names)]), name='b_out')

# W1_summary = tf.image_summary('W1_img', W1)

#########################################################
# Define layers
#########################################################

# Function to create a convolution layer, with a bias term and non-linear activation function
def conv_layer(x, W, b):
    print('x=', x)
    print('w=', W)
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    print('Conv = ', conv)
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out

# Function to create a max-pooling layer
def maxpool_layer(conv, k=2):
    return tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

#########################################################
# Function to create a full, convolutional neural network
#########################################################
def model():
    # Input layer
    x_reshaped = tf.reshape(x, shape=[-1, 24, 24, 1])

    # Construct the FIRST layer of convolution and maxpooling
    conv_out1 = conv_layer(x_reshaped, W1, b1)
    maxpool_out1 = maxpool_layer(conv_out1)
    norm1 = tf.nn.lrn(maxpool_out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    # Construct the SECOND layer of convolution and maxpooling
    conv_out2 = conv_layer(norm1, W2, b2)
    norm2 = tf.nn.lrn(conv_out2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    maxpool_out2 = maxpool_layer(norm2)

    # LASTLY, construct the concluding fully connected layers
    maxpool_reshaped = tf.reshape(maxpool_out2, [-1, W3.get_shape().as_list()[0]])
    local = tf.add(tf.matmul(maxpool_reshaped, W3), b3)
    local_out = tf.nn.relu(local)

    # Get and return output
    out = tf.add(tf.matmul(local_out, W_out), b_out)
    return out

model_op = model()
print(model)

#########################################################
# Measuring performance
#########################################################
''' With a neural network architecture designed, the next step is to define a cost function
that we wish to minimize. We'll use TensorFlow's function called softmax_cross_entropy_with_logits_v2.
The function softmax_cross_entropy_with_logits measures the probability error in discrete
classification tasks in which the classes are mutually exclusive (each entry is in exactly one
class). For example, each CIFAR-10 image is labeled with one and only one label: an image can be
a dog or a truck, but not both. Because an image can belong to 1 of 10 possible labels, we will
represent that choice as a 10-dimensional vector. All elements of this vector have a value 0,
except the element corresponding to the label will have a value of 1. This representation, as
we've seen in the earlier chapters, is called one-hot encoding.
'''

# Define the cost function using softmax_cross_entropy_with_logits_v2()
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_op, labels=y))
tf.summary.scalar('cost', cost)

# Define the training op to minimise the cost function
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Check if prediction is correct
correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

merged = tf.summary.merge_all()

#########################################################
# Train the classifier 
#########################################################
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('summaries/train', sess.graph)
    sess.run(tf.global_variables_initializer())
    onehot_labels = tf.one_hot(labels, len(names), on_value=1., off_value=0., axis=-1)
    onehot_vals = sess.run(onehot_labels)
    batch_size = len(data) / 200
    print('batch size', batch_size)
    for j in range(0, 1000):
        print('EPOCH', j)
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size, :]
            batch_onehot_vals = onehot_vals[i:i+batch_size, :]
            _, accuracy_val, summary = sess.run([train_op, accuracy, merged], feed_dict={x: batch_data, y: batch_onehot_vals})
            summary_writer.add_summary(summary, i)
            if i % 1000 == 0:
                print(i, accuracy_val)
        print('DONE WITH EPOCH')