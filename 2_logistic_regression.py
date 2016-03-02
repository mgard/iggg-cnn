import tensorflow as tf
import numpy as np

from cifarutils import loadCifar
from matplotlib import pyplot as plt

BATCH_SIZE = 256

X_train, y_train, X_valid, y_valid, X_test, y_test = loadCifar()

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


trX, trY, teX, teY = X_train.reshape(len(X_train), -1), y_train, X_test.reshape(len(X_test), -1), y_test

X = tf.placeholder("float", [None, 3072]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w = init_weights([3072, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute mean cross entropy (softmax is applied internally)
summ1 = tf.scalar_summary("Cross entropy", cost)

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression
#summ2 = tf.scalar_summary("Precision", predict_op)


sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/cifar_logs", sess.graph_def)
init = tf.initialize_all_variables()
sess.run(init)

loss, trainscore, testscore = [], [], []
for i in range(100):
    feed = {X: trX, Y: trY}
    result = sess.run([merged], feed_dict=feed)
    summary_str = result[0]
    #acc = result[1]
    writer.add_summary(summary_str, i)
    
    #loss.append(sess.run(cost, feed_dict={X: trX, Y: trY}))
    #print(">>>", loss[-1])
    #trainscore.append(np.mean(np.argmax(trY, axis=1) ==
    #                 sess.run(predict_op, feed_dict={X: trX, Y: trY})))
    #print("***", trainscore[-1])
    #testscore.append(np.mean(np.argmax(teY, axis=1) ==
    #                 sess.run(predict_op, feed_dict={X: teX, Y: teY})))
    #print (i, testscore[-1])
    for start, end in zip(range(0, len(trX), BATCH_SIZE), range(BATCH_SIZE, len(trX), BATCH_SIZE)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
