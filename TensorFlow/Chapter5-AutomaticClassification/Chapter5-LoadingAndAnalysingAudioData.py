# Note that this scipt does not seem to work in its current form. Specifically,
# there seems to be any issue with the Bregman Toolbox

''' Clustering is the process of intelligently categorizing the items in your dataset.
The overall idea is that two items in the same cluster are "closer" to each other than
items that belong to separate clusters. That is the general definition, leaving the
interpretation of "closeness" open.

    In important step before starting a complex learning algorithm is to import your
data. In this script, we will investigate how to read audio files as input to our
clustering algorithm so we automatically group together music that sounds similar.
You can use a variety of python libraries to load files onto memory, such as Numpy 
or Scipy. However, here, we will try to use TensorFlow for both the data pre-
processing as well as the learning.

    TensorFlow provides an operator to list files in a directory called
tf.train.match_filenames_once(...). We can then pass this information along to a
queue operator tf.train.string_input_producer(...). That way, we can access a
filename one at a time, without loading everything at once. Given a filename,
we can decode the file to retrieve usable data.

    These functions will allow us to load and analyse our audio data. However,
when inputing this data into a learning algorithm, it is important that we
pre-process the data and input a simplified feature vector containing only
important features of our data. With audio data, one way to create this kind
of feature vector is by using spectrograms (i.e. graphs with time on the x-axis,
frequency on the y-axis, and the color of highlighting indicating the presence/
strength of activity at a given point in time-frequency space). We could, for
example, divide the spectrogram into 100ms segements, determine the strongest
pitch in each segment, and use a histogram of these pitches as an input feature.
(i.e. with prevalence on the y-axis, and pitch on the x-axis). It is important
that this histogram is normalised (i.e. all bars adding up to 1) to allow
comparisons between audio clips of different lengths. This process in done
in the script below using the extract_feature_vector() function.
'''

# Import the usual libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from bregman.suite import *

# Define data sources, mechanisims for loading. Then load
filenames = tf.train.match_filenames_once('./audio_dataset/*.wav') # List files in the directory
count_num_files = tf.size(filenames) # Get number of files
filename_queue = tf.train.string_input_producer(filenames) # Pass this information along to a queue operator
reader = tf.WholeFileReader() # Initialise a reader (which outputs the entire contents of a file as a value)
filename, file_contents = reader.read(filename_queue) # Run the reader to extract file data

# Learning hyper-parameters
k = 2
max_iterations = 100

# Function to retrive next audio file
def get_next_chromogram(sess):
    audio_file = sess.run(filename)
    F = Chromagram(audio_file, nfft=16384, wfft=8192, nhop=2205)
    return F.X, audio_file

# Find the peak pitch for each segment sample, and create histogram of pitch counts
chromo = tf.placeholder(tf.float32) # Placeholder for audio data
max_freqs = tf.argmax(chromo, 0) # Operation to find maximum values in audio data
def extract_feature_vector(sess, chromo_data):
    num_features, num_samples = np.shape(chromo_data) # Get number of features and sample for each segment
    freq_vals = sess.run(max_freqs, feed_dict={chromo: chromo_data}) # Run max_freqs to return strongest pitches in segments
    hist, bins = np.histogram(freq_vals, bins=range(num_features + 1)) # Create histogram
    normalized_hist = hist.astype(float) / num_samples # Normalise histogram
    return normalized_hist

#
def get_dataset(sess):
    num_files = sess.run(count_num_files)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    xs = list()
    names = list()
    plt.figure()
    for _ in range(num_files):
        chromo_data, filename = get_next_chromogram(sess)

        plt.subplot(1, 2, 1)
        plt.imshow(chromo_data, cmap='Greys', interpolation='nearest')
        plt.title('Visualization of Sound Spectrum')

        plt.subplot(1, 2, 2)
        freq_vals = sess.run(max_freqs, feed_dict={chromo: chromo_data})
        plt.hist(freq_vals)
        plt.title('Histogram of Notes')
        plt.xlabel('Musical Note')
        plt.ylabel('Count')
        plt.savefig('{}.png'.format(filename))
        plt.clf()

        plt.clf()
        names.append(filename)
        x = extract_feature_vector(sess, chromo_data)
        xs.append(x)
    xs = np.asmatrix(xs)
    return xs, names


def initial_cluster_centroids(X, k):
    return X[0:k, :]


def assign_cluster(X, centroids):
    expanded_vectors = tf.expand_dims(X, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    return mins


def recompute_centroids(X, Y):
    sums = tf.unsorted_segment_sum(X, Y, k)
    counts = tf.unsorted_segment_sum(tf.ones_like(X), Y, k)
    return sums / counts


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.initialize_all_variables())
    X, names = get_dataset(sess)
    centroids = initial_cluster_centroids(X, k)
    i, converged = 0, False
    while not converged and i < max_iterations:
        i += 1
        Y = assign_cluster(X, centroids)
        centroids = sess.run(recompute_centroids(X, Y))
    print(zip(sess.run(Y), names))

















# #################################################
# # Traversing a directory for data
# #################################################


# # Loop through data
# with tf.Session() as sess:
#     # Intialise all local variables in session
#     sess.run(tf.local_variables_initializer())

#      # Get the number of files
#     num_files = sess.run(count_num_files)

#     # Initialise the creation of a new computation thread (i.e. for while, for loops, etc.)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord) # This is usually then stopped at the end
    
#     # Loop through the data one by one
#     for i in range(num_files):
#         audio_file = sess.run(filename)
#         print(audio_file)

# #################################################
# # Traversing a directory for data
# #################################################

# # Hyper-parameters for learning algorithm
# k = 2
# max_iterations = 100

# # 
