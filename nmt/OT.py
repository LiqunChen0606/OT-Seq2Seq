import numpy as np
import tensorflow as tf


def cost_matrix(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    "Returns the cosine distance"
    #NOTE: choose cosine distance here

    x = tf.nn.l2_normalize(x, 1, epsilon=1e-12)
    y = tf.nn.l2_normalize(y, 1, epsilon=1e-12)
    tmp1 = tf.matmul(x, y, transpose_b=True)
    cos_dis = 1 - tmp1

    x_col = tf.expand_dims(x, 1)
    y_lin = tf.expand_dims(y, 0)
    res = tf.reduce_sum(tf.abs(x_col - y_lin), 2)

    return cos_dis

def IPOT(C, n, m, beta=0.5):

    # sigma = tf.scalar_mul(1 / n, tf.ones([n, 1]))
    sigma = tf.ones([m, 1]) / tf.cast(m, tf.float32)
    T = tf.ones([n, m])
    A = tf.exp(-C / beta)
    for t in range(50):
        Q = tf.multiply(A, T)
        for k in range(1):
            delta = 1 / (tf.cast(n, tf.float32) * tf.matmul(Q, sigma))
            sigma = 1 / (
                tf.cast(m, tf.float32) * tf.matmul(Q, delta, transpose_a=True))
        # pdb.set_trace()
        tmp = tf.matmul(tf.diag(tf.squeeze(delta)), Q)
        T = tf.matmul(tmp, tf.diag(tf.squeeze(sigma)))
    return T


def IPOT_distance(C, n, m):
    T = IPOT(C, n, m)
    distance = tf.trace(tf.matmul(C, T, transpose_a=True))
    return distance
