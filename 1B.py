# Usual imports
import numpy as np, tensorflow as tf

# First operand
op1 = tf.constant([[2, 6], [3, 2]])

# Second operand
op2 = tf.constant([[5],[5]])

# Operation
prod = tf.matmul(op1, op2)

# Not what we expect!
print("Result?", prod)

# We compile the graph
sess = tf.Session()

# Logs the graph and results
# This is not mandatory to get the result, but allows the use of TensorBoard
writer = tf.train.SummaryWriter("/tmp/tflogs_1B", sess.graph_def, flush_secs=10)

# Get the result
result = sess.run(prod)

# Tada!
print("Actual result!", result)
