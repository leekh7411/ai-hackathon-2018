"""

 comment written by 2eeKH , 18.04.17

"""

import tensorflow as tf
import numpy as np


def text_clf_ensemble_model(input1, input2, char_size, embedding_size, is_train, keep_prob, n_classes, model_index):
    """

    Make CNN model for each input. Each model will analyse each input data.
    And we can get 'n_classes' size of feature data(dense) from each model.
    Concat them and add final dense layer for binary classification.
    This classification result will be used for ensemble prediction.

    :param input1: vertorized data([?,N] size)
    :param input2: vertorized data([?,N] size)
    :param char_size: width of embedding weight(default 251, input as char2ver vector data)
    :param embedding_size: height of embedding weight(default 8)
    :param is_train: tensorflow placeholder obj for batch_normalization's enable or disable
    :param keep_prob: tensorflow placeholder obj for dropout rate
    :param n_classes: each input cnn model's output size (final feature size)
    :param model_index: prevent redundant usage of name scope
    :return:
    """
    outputs = []
    outputs.append(text_clf_model(input=input1, is_train=is_train, char_size=char_size,
                                  embedding_size=embedding_size, keep_prob=keep_prob, model_index=1,
                                  n_classes=n_classes,
                                  conv_filters=[
                                      [[7, 7], 32],  # 1
                                      [[5, 5], 32],  # 2
                                      [[4, 4], 32],  # 3
                                      [[3, 3], 32],  # 4
                                      [[2, 2], 32]]  # 5
                                  ))

    outputs.append(text_clf_model(input=input2, is_train=is_train, char_size=char_size,
                                  embedding_size=embedding_size, keep_prob=keep_prob, model_index=2,
                                  n_classes=n_classes,
                                  conv_filters=[
                                      [[7, 7], 32],  # 1
                                      [[5, 5], 32],  # 2
                                      [[4, 4], 32],  # 3
                                      [[3, 3], 32],  # 4
                                      [[2, 2], 32]]  # 5
                                  ))

    output_concat = tf.concat(outputs, 1)
    total_output_len = len(outputs) * n_classes
    output_expand = tf.reshape(output_concat, [-1, total_output_len])
    print(output_expand)
    # Output layer
    final_output_size = 1
    W_Fin = tf.get_variable(
        name='w-fin-%d' % model_index,
        shape=[total_output_len, final_output_size],
        initializer=tf.contrib.layers.xavier_initializer()
    )

    B_Fin = tf.Variable(tf.constant(0.1, shape=[final_output_size]), name="B-final-out")
    output = tf.nn.sigmoid(tf.matmul(output_expand, W_Fin) + B_Fin)
    print(output)

    return output


def text_clf_model(input, conv_filters, char_size, embedding_size, model_index, n_classes, is_train, keep_prob):
    """

    This model extract features of input data!

    first, embedding on input data
    second, add conv layer follow 'conv_filters'.
    >> each odd layer will added previous conv feature
    finally, connect 2 fully connected layer and final output size must be 'n_classes'

    :param input: vertorized data([?,N] size)
    :param conv_filters: list of CNN kernel shape. Let's call list's element as shape.
                         shape[0] is kernel size and shape[1] is num of kernel

    :param char_size: width of embedding weight(default 251, input as char2ver vector data)
    :param embedding_size: height of embedding weight(default 8)
    :param model_index: prevent redundant usage of name scope
    :param n_classes: model's output size (final feature size)
    :param is_train: tensorflow placeholder obj for batch_normalization's enable or disable
    :param keep_prob: tensorflow placeholder obj for dropout rate
    :return:
    """
    # specification
    fully_connect_hidden_layer_size = 256

    # x = [None, 400] data
    # Embedding Layer
    # default char size = 251
    # default embedding size = 8
    with tf.name_scope("embedding" + "-%s" % model_index):
        embedding_W = tf.Variable(
            tf.random_uniform([char_size, embedding_size], -1.0, 1.0),
            name="Embedding_W" + "-%s" % model_index
        )

        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        embedded = tf.nn.embedding_lookup(embedding_W, input)
        embedded_expand = tf.expand_dims(embedded, -1)
        print(embedded_expand)

    # embedeed_expand = [None, 400, 18, 1]

    conv = embedded_expand
    for i, shape in enumerate(conv_filters):
        with tf.name_scope("conv1-%d" % model_index + str(i + 1)):
            conv = tf.layers.conv2d(conv, shape[1], shape[0], padding="SAME")
            print(conv)
        if i % 2 == 0:
            with tf.name_scope("max_pool-%d" % model_index + str(i + 1)):
                conv = tf.layers.max_pooling2d(conv, [2, 2], [2, 2], padding="SAME")
                conv = tf.nn.relu(conv)
                print(conv)

        else:
            with tf.name_scope("conv2-%d" % model_index + str(i + 1)):
                conv2 = tf.layers.conv2d(conv, shape[1], shape[0], padding="SAME")
                conv2 = tf.layers.batch_normalization(conv2, training=is_train)
                conv2 = tf.nn.relu(conv2)
                print(conv2)
            with tf.name_scope("residual-%d" % model_index + str(i + 1)):
                conv = conv + conv2

    with tf.name_scope("final_hidden%d" % model_index):
        lh = tf.contrib.layers.flatten(conv)
        lh = tf.layers.dense(lh, fully_connect_hidden_layer_size, activation=tf.nn.relu)
        lh = tf.nn.dropout(lh, keep_prob=keep_prob)
        print(lh)

    with tf.name_scope("final_layer%d" % model_index):
        w4 = tf.Variable(tf.random_normal([fully_connect_hidden_layer_size, n_classes], stddev=0.01))
        model = tf.nn.relu(tf.matmul(lh, w4))
        print(model)

    return model

def summary_layer(layer1,layer2='None',tag='None'):
    """

    this code for print the layer

    :param layer1: layer for printed
    :param layer2: if exists use but not then print None
    :param tag: additional print comment
    :return:
    """
    print('----------------------------------------------------------')
    print(tag)
    print(layer1,"\n",layer2)
    print('----------------------------------------------------------')


def weights_and_biases(a, b):
    """

    return tensorflow's basic 'weight' and 'bias' tensor
    this function for simple code


    :param a: weight's size ('a' X 'b') must be integer
    :param b: weight's size ('a' X 'b') must be integer
    :return: tensorflow's basic 'weight' and 'bias' tensor
    """
    w = tf.Variable(tf.truncated_normal(shape=[a, b], stddev=np.sqrt(2 / a)))
    b = tf.Variable(tf.zeros([b]))
    return w, b

def conv_layer2(input1,input2,filter,filter_size,index):
    """

    return tf.conv2d layer on each input


    :param input1: tensorflow model for input1
    :param input2: tensorflow model for input2
    :param filter: shape of filter for ConvNet
    :param filter_size: number of filter for ConvNet
    :param index: prevent redundant usage of name scope
    :return: tf.conv2d layers on each input
    """


    #input1 = tf.layers.batch_normalization(input1,training=is_train)
    #input2 = tf.layers.batch_normalization(input2,training=is_train)
    with tf.name_scope("conv-%d"%index):
        input1 = tf.layers.conv2d(input1, filter_size, filter, padding="SAME", activation=tf.nn.relu)
        input2 = tf.layers.conv2d(input2, filter_size, filter, padding="SAME", activation=tf.nn.relu)
        #summary_layer(input1, input2)

    with tf.name_scope("max_pool-%d"%index):
        input1 = tf.layers.max_pooling2d(input1, [2, 2], [2, 2], padding="SAME")
        input2 = tf.layers.max_pooling2d(input2, [2, 2], [2, 2], padding="SAME")
        #summary_layer(input1, input2)

    #input1 = tf.nn.l2_normalize(input1, dim=1)
    #input2 = tf.nn.l2_normalize(input2, dim=2)
    #summary_layer(input1,input2)

    return input1, input2

def conv_layer(input, filter, filter_size, index):
    """

    return conv layer for single tensorflow input model

    :param input: tensorflow model for single input
    :param filter: shape of filter for ConvNet
    :param filter_size: number of filter for ConvNet
    :param index: prevent redundant usage of name scope
    :return: conv layer for single tensorflow input model
    """

    #input = tf.layers.batch_normalization(input, training=is_train)
    with tf.name_scope("conv-%d" % index):
        input = tf.layers.conv2d(input, filter_size, filter, padding="SAME", activation=tf.nn.relu)
        #summary_layer(input)

    with tf.name_scope("max_pool-%d" % index):
        input = tf.layers.max_pooling2d(input, [2, 2], [2, 2], padding="SAME")
        #summary_layer(input)

    #input = tf.nn.l2_normalize(input, dim=1)
    #summary_layer(input)

    return input


def text2embedding4CNN1(input,char_size,embedding_size):
    """

    return tensorflow Embedding model for single input

    :param input: [?, N] size input data
    :param char_size: width of embedding weight(default 251, input as char2ver vector data)
    :param embedding_size: height of embedding weight(default 8)
    :return: tensorflow Embedding model
    """

    # Embedding Input 1 and 2
    with tf.name_scope("embedding-1"):
        embedding_W = tf.Variable(
            tf.random_uniform([char_size, embedding_size], -1.0, 1.0),
            name="Embedding_W1"
        )
        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        embedded = tf.nn.embedding_lookup(embedding_W, input)

        expand_input = tf.expand_dims(embedded, -1)

        summary_layer(expand_input)

    return expand_input


def text2embedding4CNN(input1,input2,char_size,embedding_size):
    """

    make tensorflow embedding model for input1 and input2

    :param input1: vertorized data([?,N] size)
    :param input2: vertorized data([?,N] size)
    :param char_size: width of embedding weight(default 251, input as char2ver vector data)
    :param embedding_size: height of embedding weight(default 8)
    :return: tensorflow Embedding model on each model
    """

    # Embedding Input 1 and 2
    with tf.name_scope("embedding-1"):
        embedding_W1 = tf.Variable(
            tf.random_uniform([char_size, embedding_size], -1.0, 1.0),
            name="Embedding_W1"
        )

        embedding_W2 = tf.Variable(
            tf.random_uniform([char_size, embedding_size], -1.0, 1.0),
            name="Embedding_W2"
        )

        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        embedded1 = tf.nn.embedding_lookup(embedding_W1, input1)
        embedded2 = tf.nn.embedding_lookup(embedding_W2, input2)

        expand_input1 = tf.expand_dims(embedded1, -1)
        expand_input2 = tf.expand_dims(embedded2, -1)

        summary_layer(expand_input1, expand_input2)

    return expand_input1, expand_input2






