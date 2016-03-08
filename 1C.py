import numpy as np, tensorflow as tf

# First operand
op1 = tf.placeholder(tf.int32, shape=(4, 4))

# Second operand
op2 = tf.placeholder(tf.int32, shape=(4, 3))

# Operation
prod = tf.matmul(op1, op2)


# We compile the graph
sess = tf.Session()

# Only used for vizualisation purposes
# Write graph infos to the specified file
writer = tf.train.SummaryWriter("/tmp/tf_logs_1C", 
								sess.graph.as_graph_def(add_shapes=True))

# Get the result,
# but know we have to tell Tensorflow about the actual values we want to use!
result = sess.run(prod, feed_dict={op1: np.random.randint(-10, 10, size=(4, 4)),
									op2: np.random.randint(0, 10, size=op2._shape)})

# Tada!
print("Actual result!", result)
