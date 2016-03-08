import numpy as np, tensorflow as tf

# First operand
op1 = tf.constant(40)

# Second operand
op2 = tf.constant(2)

# Operation
totalsum = tf.add(op1, op2)

# Not what we expect!
print("Result?", totalsum)


# We compile the graph
sess = tf.Session()
# and get the result
result = sess.run(totalsum)
# Tada!
print("Actual result!", result)
