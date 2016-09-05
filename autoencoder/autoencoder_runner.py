# -*- coding:utf-8 -*-
# Created by yanlei on 16-9-5 at 上午11:47.
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from autoencoder.autoencoder_models.autoencoder import Autoencoder

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def standard_scale(x_train, x_test):
    preprocessor = prep.StandardScaler().fit(x_train)
    x_train = preprocessor.transform(x_train)
    x_test = preprocessor.transform(x_test)
    return x_train, x_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


train, test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 200
batch_size = 128
display_step = 1

autoencoder = Autoencoder(n_input=784,
                          n_hidden=200,
                          transfer_function=tf.nn.softplus,
                          optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print "Epoch:", '%04d' % (epoch + 1), \
            "cost=", "{:.9f}".format(avg_cost)

print "Total cost: " + str(autoencoder.calc_total_cost(test))
