import tensorflow as tf
import numpy as np
from scipy import optimize

from tensorflow.examples.tutorials.mnist import input_data

# read data

batch_size = 128


def init_weights(shape, wd=1e-4):
    # c.f. 2.2 of [1]
    # [1]. He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification."
    # Proceedings of the IEEE International Conference on Computer Vision. 2015.
    k, c = 3, shape[-2]
    var = tf.Variable(tf.truncated_normal(shape, stddev=np.sqrt(2.0 / (k*k*c))))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd)
        tf.add_to_collection('losses', weight_decay)
    # print var.get_shape()
    return var

def model(X, W1, W2, W3, W4, W_O, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    l2a = tf.nn.relu(tf.nn.conv2d(l1, W2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    l3a = tf.nn.relu(tf.nn.conv2d(l2, W3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, W4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)
    l4 = tf.nn.relu(tf.matmul(l3, W4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    py_x = tf.matmul(l4, W_O)
    return py_x


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

# inputs and outputs
X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# Weights
W1 = init_weights([3, 3, 1, 32], wd=1e-4)
W2 = init_weights([3, 3, 32, 64], wd=1e-4)
W3 = init_weights([3, 3, 64, 128], wd=1e-4)
W4 = init_weights([128*4*4, 625], wd=1e-4)
W_O = init_weights([625, 10], wd=1e-4)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, W1, W2, W3, W4, W_O, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-7)
grads = optimizer.compute_gradients(cost)
print(grads)

with tf.Session() as sess:

    tf.global_variables_initializer().run()

    training_batch = zip(range(0, len(trX), batch_size),
                         range(batch_size, len(trX)+1, batch_size))

    for start, end in training_batch:
        sess.run(grads, feed_dict={
            X: trX[start:end],
            Y: trY[start:end],
            p_keep_conv: 0.5,
            p_keep_hidden: 0.6})

    W1 = tf.reshape(W1, [-1, 1])
    print(tf.shape(W1))
    #print(W1.eval())
    ##XX = tf.pack([W1, W2, W3], axis=2)
    ##print(XX)
