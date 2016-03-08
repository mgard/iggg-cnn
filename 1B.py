import numpy as np, tensorflow as tf

# First operand
op1 = tf.constant([[2, 6], [3, 2]])

# Second operand
op2 = tf.constant([[5],[5]])

# Operation
prod = tf.matmul(op1, op2)


# We compile the graph
sess = tf.Session()

# Only used for vizualisation purposes
# Write graph infos to the specified file
writer = tf.train.SummaryWriter("/tmp/tf_logs_1B", 
								sess.graph.as_graph_def(add_shapes=True))

# Get the result
result = sess.run(prod)

# Tada!
print("Actual result!", result)
