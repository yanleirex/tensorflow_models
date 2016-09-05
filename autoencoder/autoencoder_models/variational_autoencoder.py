# -*- coding:utf-8 -*-
# Created by yanlei on 16-9-5 at 上午11:30.def _initialize_weights(self):
import tensorflow as tf
import numpy as np
from .. import utils


class VariationalAutoencoder(object):
    def __init__(self, n_input, n_hidden, optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.z_mean = tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])
        self.z_log_sigma_sq = tf.add(tf.matmul(self.x, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])

        # sample from gaussian distribution
        eps = tf.random_normal(tf.pack([tf.shape(self.x)[0], self.n_hidden]), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean, tf.matmul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])

        # cost
        reconstruct_loss = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq),
            1)
        self.cost = tf.reduce_mean(reconstruct_loss + latent_loss)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        _weights = dict()
        _weights['w1'] = tf.Variable(utils.xavier_init(self.n_input, self.n_hidden))
        _weights['log_sigma_w1'] = tf.Variable(utils.xavier_init(self.n_input, self.n_hidden))
        _weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        _weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        _weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        _weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return _weights

    def partial_fit(self, x_):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: x_})
        return cost

    def calc_total_cost(self, x_):
        return self.sess.run(self.cost, feed_dict={self.x: x_})

    def transform(self, x_):
        return self.sess.run(self.z_mean, feed_dict={self.x: x_})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.z_mean: hidden})

    def reconstruct(self, x_):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x_})

    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_biases(self):
        return self.sess.run(self.weights['b1'])
