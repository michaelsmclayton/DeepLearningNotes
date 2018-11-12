# Import dependencies
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sess = tf.InteractiveSession() # Start an interactive session

#########################################################################
print('          Defining, updating, and saving variables')
#########################################################################

''' When covering the basics, we defined only constants. However, most
interesting applications will require data to change. For this, we will
use VARIABLES. We will use the example of neural spikes rates'''

# Define example neural data
neuralData = [1., 2., 8., -1., 0., 5.5, 6., 13]

# Create a variable (using tf.Variable) containing a vector of Boolean values that captures the history of spikes
spikes = tf.Variable([False] * len(neuralData), name='spikes')

# All variables must be initialised. This is done by calling 'run()' on its 'initializer'
spikes.initializer.run()

# Create a saver op, which enables saving and restoring variables
saver = tf.train.Saver()

# Loop through the data
for i in range(1, len(neuralData)):
    if neuralData[i] - neuralData[i-1] > 5: # When there is positive change of greater than 5...
        # Get the contents of spikes, and update as true at the current index
        newSpikes = spikes.eval()
        newSpikes[i] = True

        # To update a variable, assign it a new value using tf.assign(<var name>, <new value>).
        updater = tf.assign(spikes, newSpikes)
        
        # Evaluate the updater to see the change.
        updater.eval()

save_path = saver.save(sess, "data/spikes.ckpt")
print("spikes data saved in file: %s" % save_path)

print('')
#########################################################################
print('                     Loading variables')
#########################################################################

''' You will notice, when saving data in this way, a couple files generated in
the same directory as your source code. One of these files is spikes.ckpt. It
is a compactly stored binary file, so you cannot easily modify it with a text
editor. To retrieve this data, you can use the restore function from the saver
op '''

loadedSpikes = tf.Variable([False]*8, name='spikes') # Create a variable of the same size and name as the saved data
# spikes.initializer.run() # You no longer need to initialize this variable because it will be directly loaded
# saver = tf.train.Saver() # Create the saver op to restore saved data (not needed here, as already done above)
saver.restore(sess, "data/spikes.ckpt") # Restore data from the "spikes.ckpt" file
print(spikes.eval()) # Print the loaded data

# Remember to close the session after it will no longer be used
sess.close()