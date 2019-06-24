import numpy as np
import tensorflow as tf
from functools import partial
import pdb
def cost_matrix(x, y):
    "Returns the matrix of $|x_i-y_j|^p$."
    "Returns the cosine distance"
    #NOTE: cosine distance and Euclidean distance
    # x_col = x.unsqueeze(1)
    # y_lin = y.unsqueeze(0)
    # c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    # return c
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


def IPOT_np(C, beta=0.5):
        
    n, m = C.shape[0], C.shape[1]
    sigma = np.ones([m, 1]) / m
    T = np.ones([n, m])
    A = np.exp(-C / beta)
    for t in range(20):
        Q = np.multiply(A, T)
        for k in range(1):
            delta = 1 / (n * (Q @ sigma))
            sigma = 1 / (m * (Q.T @ delta))
        # pdb.set_trace()
        tmp = np.diag(np.squeeze(delta)) @ Q
        T = tmp @ np.diag(np.squeeze(sigma))
    return T

def IPOT_distance(C, n, m):
    T = IPOT(C, n, m)
    distance = tf.trace(tf.matmul(C, T, transpose_a=True))
    return distance


def shape_list(x):
   """Return list of dims, statically where possible."""
   x = tf.convert_to_tensor(x)
   # If unknown rank, return dynamic shape
   if x.get_shape().dims is None:
       return tf.shape(x)
   static = x.get_shape().as_list()
   shape = tf.shape(x)
   ret = []
   for i in range(len(static)):
       dim = static[i]
       if dim is None:
           dim = shape[i]
       ret.append(dim)
   return ret

def IPOT_distance2(C, beta=1, t_steps=10, k_steps=1):
   b, n, m = shape_list(C)
   sigma = tf.ones([b, m, 1]) / tf.cast(m, tf.float32)  # [b, m, 1]
   T = tf.ones([b, n, m])
   A = tf.exp(-C / beta)  # [b, n, m]
   for t in range(t_steps):
       Q = A * T  # [b, n, m]
       for k in range(k_steps):
           delta = 1 / (tf.cast(n, tf.float32) * tf.matmul(Q, sigma))  # [b, n, 1]
           sigma = 1 / (tf.cast(m, tf.float32) * tf.matmul(Q, delta, transpose_a=True))  # [b, m, 1]
       T = delta * Q * tf.transpose(sigma, [0, 2, 1])  # [b, n, m]
   distance = tf.trace(tf.matmul(C, T, transpose_a=True))
   return distance