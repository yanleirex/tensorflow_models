# -*- coding:utf-8 -*-
# Created by yanlei on 16-9-5 at 上午10:30.
import tensorflow as tf
import numpy as np
from .. import utils


class Autoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weigths = dict()
        all_weigths['w1'] = tf.Variable(utils.xavier_init(self.n_input, self.n_hidden))
        all_weigths['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weigths['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weigths['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weigths

    def partial_fit(self, x_):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: x_})
        return cost

    def calc_total_cost(self, x_):
        return self.sess.run(self.cost, feed_dict={self.x: x_})

    def transform(self, x_):
        return self.sess.run(self.hidden, feed_dict={self.x: x_})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, x_):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x_})

    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_biases(self):
        return self.sess.run(self.weights['b1'])
