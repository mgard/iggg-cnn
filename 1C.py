# Usual imports
import numpy as np, tensorflow as tf

# First operand
op1 = tf.placeholder(tf.int32, shape=(4, 4), name="InputData")

# Second operand
op2 = tf.placeholder(tf.int32, shape=(4, 3), name="OtherData")

# Operation
prod = tf.matmul(op1, op2)




# We compile the graph
sess = tf.Session()

# Logs the graph and results
# This is not mandatory to get the result, but allows the use of TensorBoard
writer = tf.train.SummaryWriter("/tmp/tflogs_1C", sess.graph_def, flush_secs=10)

# Get the result, but now we have to tell Tensorflow about the actual values we want to use!
result = sess.run(prod, feed_dict={op1: np.random.randint(-10, 10, size=(4, 4)),
									op2: np.random.randint(0, 10, size=op2._shape)})
# Tada!
print("Actual result!", result)
