# Introduction to autoencoders
''' An autoencoder is a type of neural network that tries to learn parameters
that make the output as close to the input as possible. An obvious way to do
thus is simply return the original input as the output. However, an autoencoder
is more interesting than that in that it contains a hidden layer between the
inputs and outputs that is smaller than the input. The hidden layer is therefore
a compression of your data (called ENCODING). The process of reconstructing the
input from the hidden layer is called DECODING.

    It makes sense to use object-oriented programming to design an autoencoding.
That way, we can later reuse the class in other applications. In fact, creating
our code using classes helps build deeper architectures, such as a STACKED
AUTOENCODER, which has been known to perform better empirically. This class
is defined in "autoencoder.py".

    In this class, we use BATCH TRAINING. Training a network one-by-one is the
safest bet if you are not pressured with time. But if your network is taking longer
than desired, one solution is to train it with multiple data inputs at a time, called
(i.e. batch training). Typically, as the batch size increases, the algorithm speeds
up, but has a lower likelihood of successfully converging. It is a double-edged sword.
'''

# Working with image data
''' Most neural networks, like our autoencoder, only accept one-dimensional input.
Pixels of an image, on the other hand, are indexed by both rows and columns. Furthermore,
colour images have red, green, and blue channel information for each pixel. A convenient
way to manage the higher dimensions of an image involves two steps:

    1. Convert the image to grayscale: merge the values of red, green, and blue into what
    is called the pixel intensity, which is a weighted average of the color values.

    2. Rearrange the image into row-major order. Row-major order stores an array as a longer,
    single dimension set where we just put all the dimensions of an array onto the end of the
    first dimension, and is well supported by NumPy. This allows us to index the image by 1
    number instead of 2. If an image is 3 by 3 pixels in size, we rearrange into a single vector
    of length 9.

    The code belows shows how to perform these kinds of operations when using images
with neural networks.
'''

# Different types of autoencoders
''' This chapter introduces the most straightforward type of autoencoder, but other variants
have been developed, each with their benefits and applications.

    1. A STACKED AUTOENCODER starts the same way a normal autoencoder does. It learns the encoding
    for an input into a smaller hidden layer by minimizing the reconstruction error. The hidden layer
    is now treated as the input to a new autoencoder that tries to encode the first layer of hidden
    neurons to an even smaller layer (the second layer of hidden neurons). This continues as desired.
    Often, the learned encoding weights are used as initial values for solving regression or classification
    problems in a deep neural network architecture.

    2. A DENOISING AUTOENCODER receives a noised-up input instead of the original input, and it tries to
    "denoise" it. The cost function is no longer used to minimize the reconstruction error. Now, we are
    trying to minimize the error between the denoised image and the original image. The intuition is that
    our human minds can still comprehend a photograph even after scratches or markings over it. If a machine
    can also see through the noised input to recover the original data, maybe it has a better understanding
    of the data. Denoising models have shown to better capture salient features on an image.

    3. A VARIATIONAL AUTOENCODER can generate new natural images given the hidden variables directly. Lets
    say you encode a picture of a man as a 100-dimensional vector, and then a picture of a woman as another
    100-dimensional vector. You can take the average of the two vectors, run it through the decoder, and
    produce a reasonable image that represents visually a person that is between a man and woman. This
    generative power of the variational autoencoder is derived from a type of probabilistic models called
    Bayesian networks.
'''

# Import the usual dependencies
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from matplotlib import pyplot as plt
import cPickle
import numpy as np
from autoencoder import Autoencoder

# Define some helper functions to load and preprocess the data
def grayscale(a):
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

# Reading all CIFAR-10 files to memory
names = unpickle('../cifar-10-batches-py/batches.meta')['label_names']
data, labels = [], []
for i in range(1, 6): # 1-5
    filename = '../cifar-10-batches-py/data_batch_' + str(i)
    batch_data = unpickle(filename)
    if len(data) > 0: # The rows of a data sample represent each sample, so we stack it vertically
        data = np.vstack((data, batch_data['data']))
        labels = np.vstack((labels, batch_data['labels']))
    else: # Labels are simply 1-dimensional, so we will stack them horizontally
        data = batch_data['data']
        labels = batch_data['labels']
data = grayscale(data) # Convert CIFAR-10 image to grayscale

# Collect all images of a certain class, such as "horse"
x = np.matrix(data)
y = np.array(labels)
horse_indices = np.where(y == 7)[0]
horse_x = x[horse_indices]
print(np.shape(horse_x))  # (5000, 3072)

# Train the autoencoder on images of horses
input_dim = np.shape(horse_x)[1]
hidden_dim = 100
ae = Autoencoder(input_dim, hidden_dim)
ae.train(horse_x)

# Test the autoencoder on other images
test_data = unpickle('../cifar-10-batches-py/test_batch')
test_x = grayscale(test_data['data'])
test_labels = np.array(test_data['labels'])
encodings = ae.classify(test_x, test_labels)

# Decode images
numberOfImages = 4
plt.rcParams['figure.figsize'] = (100, 100)
plt.figure()
for i in range(numberOfImages):
    plt.subplot(numberOfImages, 2, i*2 + 1)
    original_img = np.reshape(test_x[i, :], (32, 32))
    plt.imshow(original_img, cmap='Greys_r')
    
    plt.subplot(numberOfImages, 2, i*2 + 2)
    reconstructed_img = ae.decode([encodings[i]])
    plt.imshow(reconstructed_img, cmap='Greys_r')

plt.show()