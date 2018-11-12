# Import dependencies
import tensorflow as tf
import numpy as np

# Prevent display of warnings about how TensorFlow was compiled
'''The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#########################################################################
print('         Different ways of representing matrices')
#########################################################################

####################################
# Define the same matrix in three different ways
####################################

# As a list
matrix1 = [[1.0, 2.0],
      [3.0, 4.0]]
print(type(matrix1)) # <type 'list'>

# As an ndarray (using np.array from the NumPy library)
matrix2 = np.array([[1.0, 2.0],
               [3.0, 4.0]], dtype=np.float32)
print(type(matrix2)) # <type 'numpy.ndarray'>

# As TensorFlow's constant Tensor object (using tf.constant())
matrix3 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(type(matrix3)) # <class 'tensorflow.python.framework.ops.Tensor'>


####################################
# Create tensor objects out of the various different types (i.e. convert them)
####################################
t1 = tf.convert_to_tensor(matrix1, dtype=tf.float32)
t2 = tf.convert_to_tensor(matrix2, dtype=tf.float32)
t3 = tf.convert_to_tensor(matrix3, dtype=tf.float32)
print(type(t1)) # <class 'tensorflow.python.framework.ops.Tensor'>
print(type(t2)) # <class 'tensorflow.python.framework.ops.Tensor'>
print(type(t3)) # <class 'tensorflow.python.framework.ops.Tensor'>
                # ^^ notice that all of the above variables have the same type

####################################
# Create homogenous tensors (i.e. all zeroes, or all ones)
####################################
tensorShape = tf.constant([2, 2])
allZeros = tf.zeros(tensorShape)
allOnes = tf.ones(tensorShape)


print(' ')
#########################################################################
print('                     Creating operators')
#########################################################################

x = tf.constant([1.0, 2.0])
y = tf.constant([3.0, 4.0])

tf.negative(x) # Negate a tensor
tf.add(x, y) # Add two tensors of the same type, x + y
tf.subtract(x, y) # Subtract tensors of the same type, x - y
tf.multiply(x, y) # Multiply two tensors element-wise
tf.pow(x, y) # Take the element-wise power of x to y
tf.exp(x) # Equivalent to pow(e, x), where e is Eulers number (2.718...)
tf.sqrt(x) # Equivalent to pow(x, 0.5)
tf.div(x, y) # Take the element-wise division of x and y
tf.truediv(x, y) # Same as tf.div, except casts the arguments as a float
tf.floordiv(x, y) # Same astruediv, except rounds down the final answer into an integer
tf.mod(x, y) # Takes the element-wise remainder from division


print(' ')
#########################################################################
print('             Executing operators with sessions')
#########################################################################

''' To execute an operation and retrieve its calculated value, TensorFlow
requires a session. Only a registered session may fill the values of a Tensor
object. To do so, you must create a session class using tf.Session() and tell
it to run an operator. Not only does a session configure where your code will
be computed on your machine, but it also crafts how the computation will be laid
out in order to parallelize computation. Sessions are essential in TensorFlow code.
You need to call a session to actually "run" the math
'''

####################################
# Starting a session
####################################

# Start a session using tf.Session()
with tf.Session() as sess: # Start a session
    allZeroValues = sess.run(allZeros) # Use sess.run to evaluate an operator
    divisionResult = sess.run(tf.div(x, y))
print(allZeroValues)
print(divisionResult)

# Start a session using tf.InteractiveSession()
sess = tf.InteractiveSession()
allOneValues = allOnes.eval() # Use the Tensor.eval() function to evaluate operator
print(allOneValues)
sess.close() # Remember to close the session to free up resources


####################################
# Session configurations
####################################
''' TensorFlow automatically determines the best way to assign a GPU or CPU device
to an operation. We can pass an additional option to tf.Session, log_device_placements=True,
when creating a Session. This will show you exactly where on your hardware the
computations are evoked. More specifically, this outputs info about which CPU/GPU
devices are used in the session for each operation.
'''
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    negX = sess.run(tf.negative(x))
print(negX) # /job:localhost/replica:0/task:0/cpu:0


####################################
# Inputs to a session
####################################
''' In general, TensorFlow session take three different types of input:

    PLACEHOLDERS. A value that is unassigned, but will be initialized
    by the session wherever it is run. Typically, placeholders are the
    input and output of your model.

    VARIABLES. A value that can change, such as parameters of a machine
    learning model. Variables must me initialized by the session before
    they are used.

    CONSTANTS - A value that does not change, such as hyper-parameters
    or settings.
'''


####################################
# Understanding code as a graph
####################################
'''  It is important when learning TensorFlow to think about multiple steps
of computation as steps along a graph, or network. Each operation in the graph
is a node, and the incoming/outgoing edges of this node are how the Tensor
transforms. A tensor flows through the graph, which is why this library is
called TensorFlow! These graphs, in whole, represent complicated mathematical
functions. However, these graphs are made up of small segments, representing
simple mathematical concepts (such as negation or doubling).

    TensorFlow algorithms are easy to visualize. They can be simply described
by flowcharts. The technical (and more correct) term for such a flowchart is a
dataflow graph. Every arrow in a dataflow graph is called the edge. In addition,
every state of the dataflow graph is called a node. The purpose of the session
is to interpret your Python code into a dataflow graph, and then associate the
computation of each node of the graph to the CPU or GPU.
'''