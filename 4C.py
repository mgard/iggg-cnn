# Usual imports
import tensorflow as tf, numpy as np, time
from cifarutils import loadCifar

BATCH_SIZE = 256
NUM_EPOCHS = 250

# Fetch dataset and reshape it
X_train, y_train, X_valid, y_valid, X_test, y_test = loadCifar()
trX, trY, teX, teY = X_train.transpose([0, 2, 3, 1]), y_train, X_test.transpose([0, 2, 3, 1]), y_test

# Create input and output nodes
X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

# Create our weights matrix (and provide initialization info)
w_c1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.2))
w_c2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.2))
w_c3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.2))
w_f1 = tf.Variable(tf.truncated_normal([128 * 4 * 4, 512], stddev=0.2))
b_f1 = tf.Variable(tf.zeros([512]))
w_out = tf.Variable(tf.truncated_normal([512, 10], stddev=0.2))

# We use these placeholder to parametrize the dropout ratio
dropout_rate_conv = tf.placeholder("float")
dropout_rate_full = tf.placeholder("float")

# Define our model (how do we predict)
pred = tf.nn.elu(tf.nn.conv2d(X, w_c1, strides=[1, 1, 1, 1], padding='SAME'))
pred = tf.nn.max_pool(pred, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
pred = tf.nn.dropout(pred, 1.-dropout_rate_conv)

pred = tf.nn.elu(tf.nn.conv2d(pred, w_c2, strides=[1, 1, 1, 1], padding='SAME'))
pred = tf.nn.max_pool(pred, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
pred = tf.nn.dropout(pred, 1.-dropout_rate_conv)

pred = tf.nn.elu(tf.nn.conv2d(pred, w_c3, strides=[1, 1, 1, 1], padding='SAME'))
pred = tf.nn.max_pool(pred, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
pred = tf.nn.dropout(pred, 1.-dropout_rate_conv)

pred = tf.reshape(pred, [-1, 128 * 4 * 4])
pred = tf.nn.elu(tf.matmul(pred, w_f1) + b_f1)
pred = tf.nn.dropout(pred, 1.-dropout_rate_full)
pred = tf.matmul(pred, w_out)


# Define the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
# Use a gradient descent as optimization method
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

# This will be useful to log accuracy
# We define the prediction as the index of the highest output
predict_op = tf.argmax(pred, 1)
gndtruth_op = tf.argmax(Y, 1)

# This node checks if the prediction is equal to the actual answer
ispredcorrect = tf.equal(predict_op, gndtruth_op)
accuracy = tf.reduce_mean(tf.cast(ispredcorrect, 'float'))

# Only used for vizualisation purposes
loss_disp = tf.scalar_summary("Cross entropy", loss)
w_disp = tf.histogram_summary("W (conv layer #1)", w_c1)
w_disp2 = tf.histogram_summary("W (conv layer #2)", w_c2)
w_disp3 = tf.histogram_summary("W (conv layer #3)", w_c3)
w_disp4 = tf.histogram_summary("W (hidden layer #4)", w_f1)
w_disp5 = tf.histogram_summary("W (output layer)", w_out)
acc_disp = tf.scalar_summary("Accuracy (train)", accuracy)
merged_display = tf.merge_all_summaries()

# We also add a vizualisation of the performance on the test dataset
acc_test_disp = tf.scalar_summary("Accuracy (test)", accuracy)


# We compile the graph
sess = tf.Session()
# Write graph infos to the specified file
writer = tf.train.SummaryWriter("/tmp/tflogs_4C", sess.graph_def, flush_secs=10)

# We must initialize the values of our variables
init = tf.initialize_all_variables()
sess.run(init)


dictTrain = {X: trX, Y: trY, dropout_rate_conv: 0.2, dropout_rate_full: 0.5}
dictTest = {X: teX, Y: teY, dropout_rate_conv: 0.0, dropout_rate_full: 0.0}
for i in range(NUM_EPOCHS):
    # This is only used to fetch some interesting information
    # and plot them in tensorboard and display them in the terminal.
    # They are NOT mandatory to train the network
    begin = time.time()
    result = sess.run(merged_display, feed_dict=dictTrain)
    writer.add_summary(result, i)
    result = sess.run(acc_test_disp, feed_dict=dictTest)
    writer.add_summary(result, i)
    writer.flush()
    
    print("[{}]".format(i), end=" ")
    trainPerf = sess.run(accuracy, feed_dict=dictTrain)
    testPerf = sess.run(accuracy, feed_dict=dictTest)
    print("Train/Test accuracy : {:.4f} / {:.4f}".format(trainPerf, testPerf), end=" ")
    
    # This is the actual training
    # We divide the dataset in mini-batches
    for start, end in zip(range(0, len(trX), BATCH_SIZE),
                            range(BATCH_SIZE, len(trX), BATCH_SIZE)):
        # For each batch, we train the network and update its weights
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], dropout_rate_conv: 0.2, dropout_rate_full: 0.5})
    print("(done in {:.2f} seconds)".format(time.time()-begin))
