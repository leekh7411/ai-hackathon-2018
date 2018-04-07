import argparse
import os

import numpy as np
import tensorflow as tf
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from kin_dataset import KinQueryDataset, preprocess

def text_cnn_ver_3(_x, _character_size, _embedding_size, model_index,n_classes,keep_prob):
    # specification
    conv1_filter_shape = [5, 5]
    conv2_filter_shape = [4, 4]
    conv3_filter_shape = [3, 3]
    conv4_filter_shape = [2, 2]
    conv1_filter_num = 32  # kernel size 32
    conv2_filter_num = 32  # kernel size 64
    conv3_filter_num = 32
    conv4_filter_num = 32
    fully_connect_hidden_layer_size = 128

    # x = [None, 400] data

    # Embedding Layer
    # default char size = 251
    # default embedding size = 8
    with tf.name_scope("embedding" + "-%s" % model_index):
        embedding_W = tf.Variable(
            tf.random_uniform([_character_size, _embedding_size], -1.0, 1.0),
            name="Embedding_W" + "-%s" % model_index
        )

        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        embedded = tf.nn.embedding_lookup(embedding_W, _x)
        embedded_expand = tf.expand_dims(embedded, -1)
        print(embedded_expand)

    # embedeed_expand = [None, 400, 18, 1]
    with tf.name_scope("conv1"):
        conv1 = tf.layers.conv2d(embedded_expand, conv1_filter_num, conv1_filter_shape,padding="SAME")
        print(conv1)

    with tf.name_scope("max_pool_1"):
        conv1_max_pooled = tf.layers.max_pooling2d(conv1,[2,2],[2,2],padding="SAME")
        #conv1_max_pooled = tf.nn.dropout(conv1_max_pooled,keep_prob=keep_prob)
        print(conv1_max_pooled)

    with tf.name_scope("conv2"):
        conv2 = tf.layers.conv2d(conv1_max_pooled,conv2_filter_num, conv2_filter_shape,padding="SAME")
        print(conv2)

    with tf.name_scope("max_pool2"):
        conv2_max_pooled = tf.layers.max_pooling2d(conv2,[2,2],[2,2],padding="SAME")
        #conv2_max_pooled = tf.nn.dropout(conv2_max_pooled,keep_prob=keep_prob)
        print(conv2_max_pooled)

    with tf.name_scope("conv3"):
        conv3 = tf.layers.conv2d(conv2_max_pooled,conv3_filter_num, conv3_filter_shape,padding="SAME")
        print(conv3)

    with tf.name_scope("max_pool3"):
        conv3_max_pooled = tf.layers.max_pooling2d(conv2,[2,2],[2,2],padding="SAME")
        #conv3_max_pooled = tf.nn.dropout(conv3_max_pooled,keep_prob=keep_prob)
        print(conv3_max_pooled)

    with tf.name_scope("conv4"):
        conv4 = tf.layers.conv2d(conv3_max_pooled,conv4_filter_num, conv4_filter_shape,padding="SAME")

    with tf.name_scope("max_pool4"):
        conv4_max_pooled = tf.layers.max_pooling2d(conv4,[2,2],[2,2],padding="SAME")
        #conv4_max_pooled = tf.nn.dropout(conv4_max_pooled,keep_prob=keep_prob)

    with tf.name_scope("final_hidden"):
        lh = tf.contrib.layers.flatten(conv4_max_pooled)
        lh = tf.layers.dense(lh,fully_connect_hidden_layer_size,activation=tf.nn.relu)
        #lh = tf.nn.dropout(lh, keep_prob=keep_prob)
        print(lh)

    with tf.name_scope("final_layer"):
        w4 = tf.Variable(tf.random_normal([fully_connect_hidden_layer_size, n_classes], stddev=0.01))
        model = tf.nn.relu(tf.matmul(lh, w4))
        print(model)

    return model


def text_cnn_ver_2(_x, _character_size, _embedding_size, model_index,is_train,drop_out,n_classes):
    # specification
    conv1_filter_shape = [2, 2]
    conv2_filter_shape = [3, 3]
    conv1_filter_num = 32  # kernel size 32
    conv2_filter_num = 64  # kernel size 64
    fully_connect_hidden_layer = 64

    # x = [None, 400] data

    # Embedding Layer
    # default char size = 251
    # default embedding size = 8
    with tf.name_scope("embedding" + "-%s" % model_index):
        embedding_W = tf.Variable(
            tf.random_uniform([_character_size, _embedding_size], -1.0, 1.0),
            name="Embedding_W" + "-%s" % model_index
        )

        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        embedded = tf.nn.embedding_lookup(embedding_W, _x)
        embedded_expand = tf.expand_dims(embedded, -1)
        #embedded_expand = tf.layers.batch_normalization(embedded_expand,training=is_train)
        print(embedded_expand)

    # embedeed_expand = [None, 400, 18, 1]
    with tf.name_scope("conv1"):
        conv1 = tf.layers.conv2d(embedded_expand, conv1_filter_num, conv1_filter_shape,padding="SAME")
        #conv1 = tf.layers.batch_normalization(conv1,training=is_train)
        print(conv1)

    with tf.name_scope("max_pool_1"):
        conv1_max_pooled = tf.layers.max_pooling2d(conv1,[2,2],[2,2],padding="SAME")
        #conv1_max_pooled = tf.layers.batch_normalization(conv1_max_pooled,training=is_train)
        print(conv1_max_pooled)

    with tf.name_scope("conv2"):
        conv2 = tf.layers.conv2d(conv1_max_pooled,conv2_filter_num, conv2_filter_shape,padding="SAME")
        #conv2 = tf.layers.batch_normalization(conv2,training=is_train)
        print(conv2)

    with tf.name_scope("max_pool2"):
        conv2_max_pooled = tf.layers.max_pooling2d(conv2,[2,2],[2,2],padding="SAME")
        #conv2_max_pooled = tf.layers.batch_normalization(conv2_max_pooled,training=is_train)
        print(conv2_max_pooled)

    with tf.name_scope("layer3"):
        l3 = tf.contrib.layers.flatten(conv2_max_pooled)
        l3 = tf.layers.dense(l3,fully_connect_hidden_layer,activation=tf.nn.relu)
        l3 = tf.nn.dropout(l3, keep_prob=drop_out)
        print(l3)

    with tf.name_scope("final_layer"):
        w4 = tf.Variable(tf.random_normal([fully_connect_hidden_layer, n_classes], stddev=0.01))
        model = tf.nn.relu(tf.matmul(l3, w4))
        print(model)

    return model



# MAX 0.88...
def text_cnn_ver_1(_x, _filter_sizes, _character_size, _num_filters, _embedding_size, drop_out, model_index, strmaxlen, n_classes, is_train):
    # 모델의 specification
    # ====================================== MODEL ===========================================#
    #                           Text Classification using CNN

    l2_loss = tf.constant(0.0)

    # Embedding Layer
    # _parm: char_size = 251(default)
    # _parm: config.embedding = 8(default)
    # char_embedding is tf.Variable size[251,8]
    with tf.name_scope("embedding"+ "-%s" % model_index):
        embedding_W = tf.Variable(
            tf.random_uniform([_character_size, _embedding_size], -1.0, 1.0),
            name="Embedding_W"+ "-%s" % model_index
        )

        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        embedded = tf.nn.embedding_lookup(embedding_W, _x)
        embedded_expand = tf.expand_dims(embedded, -1)
        #embedded_expand = tf.layers.batch_normalization(embedded_expand,training=is_train)
        print(embedded_expand)

    # CNN-Clf Layer1
    # create convolution + maxpool layer
    num_of_filters = _num_filters # 256
    filter_sizes = _filter_sizes  # [2,3,4,5]
    pooled_outputs = []

    # Convolution Layer
    for filter_size in filter_sizes:
        filter_name = "conv-maxpool-1-%s" % filter_size
        with tf.name_scope(filter_name + "-%s" % model_index):

            print("Embedded_Expand%s: "%filter_size , embedded_expand)

            filter_shape = [filter_size, _embedding_size, 1, num_of_filters]
            Conv_W = tf.Variable(tf.random_normal(filter_shape, stddev=0.1), name="Conv_W-%s" % model_index)  # Conv's filter?
            Conv_B = tf.Variable(tf.constant(0.1, shape=[num_of_filters]), name="Conv_B-%s" % model_index)
            Conv = tf.nn.conv2d(
                embedded_expand,
                Conv_W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="Conv-%s" % model_index
            )
            # Add Bias and Activation Relu
            h = tf.nn.relu(tf.nn.bias_add(Conv, Conv_B), name="Conv_activation_relu-%s" % model_index)

            # Batch nomalization
            #h = tf.layers.batch_normalization(h,training=is_train)

            print("Conv+Bias %s"%filter_size, h)

            # Max pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, strmaxlen - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="MaxPool" + "-%s" % model_index
            )

            print("MaxPool%s"%filter_size , pooled)

            '''# Normalization
            norm1 = tf.nn.lrn(pooled,
                              depth_radius=5,
                              bias=1.0,
                              alpha=0.001 / 9.0,
                              beta=0.75,
                              name='norm1-%s'% model_index)

            print("Normal%s"%filter_size, norm1)'''
            '''
            # Conv
            filter_shape2 = [filter_size, 1, 1, num_of_filters]
            Conv_W2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="Conv_W-%s" % model_index)
            Conv_B2 = tf.Variable(tf.constant(0.1, shape=[num_of_filters]), name="Conv_B-%s" % model_index)
            Conv2 = tf.nn.conv2d(
                pooled,
                Conv_W2,
                strides=[1,1,1,1],
                name="Conv-%s" % model_index,
                padding="VALID"
            )
            # Add Bias and Activation Relu
            h2 = tf.nn.relu(tf.nn.bias_add(Conv2,Conv_B2), name="Conv_activation_relu-%s" % model_index)

            # Max Pooling
            pooled2 = tf.nn.max_pool(
                h2,
                ksize=[1, 400 - filter_size + 1, 1, 1],
                strides=[1,1,1,1],
                padding="VALID",
                name="MaxPool2" + "-%s" % model_index
            )'''

            pooled_outputs.append(pooled)


    # Combine all the pooled features
    num_total_filters = num_of_filters * len(filter_sizes)  # 1 -> length of filter size
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_expand = tf.reshape(h_pool, [-1, num_total_filters])

    # Add Drop out

    with tf.name_scope("dropout"+ "-%s" % model_index):
        h_drop = tf.nn.dropout(h_pool_expand, drop_out)

    # Output layer
    with tf.name_scope("output-layer"+ "-%s" % model_index):
        '''W2 = tf.get_variable(
            "W-out"+ "-%s" % model_index,
            shape=[num_total_filters, n_classes],
            initializer=tf.contrib.layers.xavier_initializer()
        )'''
        W2 = weight_variable([num_total_filters,n_classes])
        #B2 = tf.Variable(tf.constant(0.1, shape=[n_classes]), name="B-out" + "-%s" % model_index)
        B2 = bias_variable([n_classes])
        l2_loss += tf.nn.l2_loss(W2)
        l2_loss += tf.nn.l2_loss(B2)

        output = tf.nn.sigmoid(tf.matmul(h_drop, W2) + B2)
        print(output)

    return output

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def textRNNModel(input, n_hidden, n_class,_char_size,_embbed_size,model_index,is_train):
    # Embedding Layer
    # _parm: char_size = 251(default)
    # _parm: config.embedding = 8(default)
    # char_embedding is tf.Variable size[251,8]
    with tf.name_scope("embedding"):
        embedding_W = tf.Variable(
            tf.random_uniform([_char_size, _embbed_size], -1.0, 1.0),
            name="Embedding_W" + "-%s" % model_index
        )

        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        embedded = tf.nn.embedding_lookup(embedding_W, input)
        embedded_expand = tf.expand_dims(embedded, -1)
        embedded_expand = tf.layers.batch_normalization(embedded_expand, training=is_train)
        print(embedded_expand)

    with tf.name_scope("rnn-layer"):
        cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
        outputs, states = tf.nn.dynamic_rnn(cell, embedded, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]

    with tf.name_scope("Output-layer"):
        w = tf.Variable(tf.random_normal([n_hidden, n_class]),dtype=tf.float32)
        b = tf.Variable(tf.random_normal([n_class]),dtype=tf.float32)
        model = tf.nn.sigmoid(tf.matmul(outputs, w) + b)

    return model