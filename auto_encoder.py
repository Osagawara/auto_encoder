import tensorflow as tf
import numpy as np
import WhatWhereAutoencoder as wwa

from functools import reduce

class AutoEncoder:
    def __init__(self, npy_path=None, trainable=True):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(self, bgr):
        self.bn1 = self.bn_layer(bgr, "batch_norm1")
        self.conv1_1 = self.conv_layer(self.bn1, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.bn2 = self.bn_layer(self.pool1, "batch_norm2")
        self.conv2_1 = self.conv_layer(self.bn2, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        '''
        self.bn3 = tf.nn.batch_normalization(self.pool2, 0, 1, offset=None, scale=None, name="batch_norm_3")
        self.conv3_1 = self.conv_layer(self.bn3, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        '''

        self.bn4 = self.bn_layer(self.pool2, "batch_norm4")
        self.upsamle2 = wwa.upsample(self.bn4, stride=2)
        self.deconv2_1 = self.conv_layer(self.upsamle2, 128, 128, "deconv2_1")
        self.deconv2_2 = self.conv_layer(self.deconv2_1, 128, 64, "deconv2_2")

        self.bn5 = self.bn_layer(self.deconv2_2, "batch_norm5")
        self.upsamle1 = wwa.upsample(self.bn5, stride=2)
        self.deconv1_1 = self.conv_layer(self.upsamle1, 64, 64, "deconv1_1")
        self.deconv1_2 = self.conv_layer_no_relu(self.deconv1_1, 64, 3, "deconv1_2")

        self.bn6 = self.bn_layer(self.deconv1_2, "batch_norm6")


    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def conv_layer_no_relu(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

            return bias

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def bn_layer(self, bottom, name):
        '''
        global normalize the input so that the mean is zero and the variance is 1
        :param bottom: 
        :param name: 
        :return: 
        '''
        with tf.variable_scope(name):
            mean, variance = tf.nn.moments(bottom, [0, 1, 2])
            bn = tf.nn.batch_normalization(bottom, mean, variance, None, None, 10**-8, name)
            return bn

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./autoencoder-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count




