# -*- coding:utf-8 -*-
# Created by yanlei on 16-9-5 at 上午10:27.
import numpy as np
import tensorflow as tf


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)
