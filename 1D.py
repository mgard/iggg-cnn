import tensorflow as tf, numpy as np
from cifarutils import loadCifar


dataX = np.linspace(-10, 10, 100)
dataY = dataX / 7.782

X = tf.placeholder("float", [100]) # create symbolic variables
Y = tf.placeholder("float", [100])

var = tf.Variable(tf.random_normal((1,)))

predy = tf.mul(X, var)

cost = tf.reduce_mean((predy - Y)**2)

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


loss, trainscore, testscore = [], [], []
for i in range(100):
    sess.run(train_op, feed_dict={X: dataX, Y: dataY})
    loss.append(sess.run(cost, feed_dict={X: dataX, Y: dataY}))
    print(sess.run(var)[0], loss[-1])
