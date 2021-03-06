import tensorflow as tf
import numpy as np

from cifarutils import loadCifar
from matplotlib import pyplot as plt

BATCH_SIZE = 256

X_train, y_train, X_valid, y_valid, X_test, y_test = loadCifar()


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


trX, trY, teX, teY = X_train.transpose([0, 2, 3, 1]), y_train, X_test.transpose([0, 2, 3, 1]), y_test

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 3, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 * 4 * 4, 625])
w_o = init_weights([625, 10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

loss, trainscore, testscore = [], [], []
for i in range(100):
    loss.append(sess.run(cost, feed_dict={X: trX, Y: trY, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
    print(">>>", loss[-1])
    trainscore.append(np.mean(np.argmax(trY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: trX, Y: trY, p_keep_conv: 1.0, p_keep_hidden: 1.0})))
    print("***", trainscore[-1])
    testscore.append(np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0})))
    print (i, testscore[-1])
    for start, end in zip(range(0, len(trX), BATCH_SIZE), range(BATCH_SIZE, len(trX), BATCH_SIZE)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                      p_keep_conv: 0.8, p_keep_hidden: 0.5})
    
    #test_indices = np.arange(len(teX)) # Get A Test Batch
    #np.random.shuffle(test_indices)
    #test_indices = test_indices[0:256]
    
    #print i, np.mean(np.argmax(teY[test_indices], axis=1) ==
    #                 sess.run(predict_op, feed_dict={X: teX[test_indices],
    #                                                 Y: teY[test_indices],
    #                                                 p_keep_conv: 1.0,
    #                                                 p_keep_hidden: 1.0}))
