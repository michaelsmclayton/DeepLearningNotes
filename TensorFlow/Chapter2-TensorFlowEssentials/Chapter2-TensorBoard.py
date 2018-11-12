# Import dependencies
import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sess = tf.InteractiveSession() # Start an interactive session

''' In machine learning, the most time-consuming part is usually not programming,
but instead it is waiting for code to finish running. TensorFlow comes with a handy
dashboard called TensorBoard for a quick peek into how values are changing in each
node of the graph. That way, you can have an intuitive idea of how your code is
performing'''

''' In the following code, we will visualise variable trends over time in a real-
world example. Specifically, we will implement a moving-average algorithm in TensorFlow
and then carefully track the variables we care about for visualization in TensorBoard.
'''

#########################################################################
print('          Implementing and visualising a moving average')
#########################################################################

'''Suppose you are interested in calculating the average stock price of a company.
In a simple moving average, all past observations are weighted equally. However,
given that recent changes might be more important for tracking a stock, it may
be better to use exponential averaging. In this process, an exponential function
is used to assign exponentially decreasing weights over time. Alpha is a parameter
that will be tuned, representing how strongly recent values should be biased
in the calculation of the average.
'''

# Algorithm parameters
raw_data = np.random.normal(10, 1, 100) # random data
alpha = tf.constant(0.05) # Intitialise alpha as a constant (of 0.5)
curr_value = tf.placeholder(tf.float32) # Intitialise curr_value as a placeholder
prev_avg = tf.Variable(0.) # Initialise prev_avg as a variable (of type: float)

# Define the average update operator
update_avg = alpha * curr_value + (1 - alpha) * prev_avg

##################################################
# State what we want to visualise
##################################################
''' For visualising data using TensorBoard, you first need to pick out which
nodes you really care about measuring by annotating them with a summary op.
A summary op produces serialized string used by a SummaryWriter to save updates
to a directory. Every time you call the add_summary method from the SummaryWriter,
TensorFlow will save data to disk for TensorBoard to use.
'''
avg_hist = tf.summary.scalar("running_average", update_avg)
value_hist = tf.summary.scalar("incoming_values", curr_value)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs")

# Function to initialize all trainable variables in one go
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialize all trainable variables
    sess.run(init) 

    # Add the computation graph to be visualised in TensorBoard
    writer.add_graph(sess.graph)

    # Loop through the data one-by-one to update the average
    for i in range(len(raw_data)):

        # Run the merged op and the update_avg op at the same time
        summary_str, curr_avg = sess.run([merged, update_avg], feed_dict={curr_value: raw_data[i]})
        
        # Update previous average with current average
        sess.run(tf.assign(prev_avg, curr_avg)) 

        # Print current value, and current average
        print(raw_data[i], curr_avg)

        # Add data to logs
        writer.add_summary(summary_str, i) # We call add_summary on them to queue up data to be written to disk

        ''' Be careful not to call the add_summary function too often! Although doing
        so will produce higher resolution visualizations of your variables, it will be
        at the cost of more computation and slightly slower learning'''

''' We can view that TensorBoard by entering the following command at the terminal'''
# tensorboard --logdir=./logs