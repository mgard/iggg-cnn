import tensorflow as tf
import numpy as np

from cifarutils import loadCifar
from matplotlib import pyplot as plt

BATCH_SIZE = 256

X_train, y_train, X_valid, y_valid, X_test, y_test = loadCifar()

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us


trX, trY, teX, teY = X_train.reshape(len(X_train), -1), y_train, X_test.reshape(len(X_test), -1), y_test

X = tf.placeholder("float", [None, 3072]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([3072, 768]) # create symbolic variables
w_o = init_weights([768, 10])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

loss, trainscore, testscore = [], [], []
for i in range(100):
    loss.append(sess.run(cost, feed_dict={X: trX, Y: trY}))
    print(">>>", loss[-1])
    trainscore.append(np.mean(np.argmax(trY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: trX, Y: trY})))
    print("***", trainscore[-1])
    testscore.append(np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY})))
    print (i, testscore[-1])
    for start, end in zip(range(0, len(trX), BATCH_SIZE), range(BATCH_SIZE, len(trX), BATCH_SIZE)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    #print i, np.mean(np.argmax(teY, axis=1) ==
    #                 sess.run(predict_op, feed_dict={X: teX, Y: teY}))
