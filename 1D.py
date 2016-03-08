# Usual imports
import tensorflow as tf, numpy as np

NUM_EPOCHS = 100

# Create dataset
dataX = np.linspace(-10, 10, 100)
dataY = dataX / 7.782

# Create input and output nodes
X = tf.placeholder("float", [100], name="X")
Y = tf.placeholder("float", [100], name="Y")

# Create the variable we want to optimize
var = tf.Variable(tf.random_normal([1]), name="multiplicator")

# Build the graph
predy = tf.mul(X, var)

# Get the loss (here, a simple SSE)
loss = tf.reduce_mean((predy - Y)**2)

# We use a built-in optimizer (Gradient descent)
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
cost_display = tf.scalar_summary("Loss", loss)
mult_display = tf.scalar_summary("a", var.value()[0])

# We compile the graph
sess = tf.Session()


# Only used for vizualisation purposes
# Write graph infos to the specified file
merged_display = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/tflogs_1D", sess.graph_def, flush_secs=10)

# We must initialize the values of our variables
init = tf.initialize_all_variables()
sess.run(init)

for i in range(NUM_EPOCHS):
    # This will train the network using our dataset
    sess.run(train_op, feed_dict={X: dataX,
                                  Y: dataY})
    # Only for debug / display purposes
    lval = sess.run(merged_display, feed_dict={X: dataX,
                                             Y: dataY})
    writer.add_summary(lval, i)
