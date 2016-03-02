import numpy as np
import tensorflow as tf

# First operand
op1 = tf.constant(40)

# Second operand
op2 = tf.constant(2)

# Operation
totalsum = tf.add(op1, op2)

# Does not do what we expect!
print(totalsum)


# We compile the graph
sess = tf.Session()
# and get the result
result = sess.run(totalsum)
# Tada!
print(result)
