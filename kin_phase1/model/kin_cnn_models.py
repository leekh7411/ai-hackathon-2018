'''
    old version models
    keep for reuse
'''
import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS
BN_EPSILON = 0.001
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')

def summary_layer(layer1,layer2='None',tag='None'):
    print('----------------------------------------------------------')
    print(tag)
    print(layer1,"\n",layer2)
    print('----------------------------------------------------------')


def weights_and_biases(a, b):
    w = tf.Variable(tf.truncated_normal(shape=[a, b], stddev=np.sqrt(2 / a)))
    b = tf.Variable(tf.zeros([b]))
    return w, b

def conv_layer2(input1,input2,filter,filter_size,index):
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

def text_clf_ensemble_model(input1,input2,char_size,embedding_size,is_train,keep_prob,n_classes,model_index):
    outputs = []
    outputs.append(text_clf_model_ver1(input=input1, is_train=is_train, char_size=char_size,
                                       embedding_size=embedding_size, keep_prob=keep_prob, model_index=1,
                                       n_classes=n_classes,
                                       conv_filters=[
                                           [[7, 7], 16],  # 1
                                           [[5, 5], 16],  # 2
                                           [[4, 4], 16],  # 3
                                           [[3, 3], 16],  # 4
                                           [[2, 2], 16]]  # 5
                                       ))

    outputs.append(text_clf_model_ver1(input=input2, is_train=is_train, char_size=char_size,
                                       embedding_size=embedding_size, keep_prob=keep_prob, model_index=2,
                                       n_classes=n_classes,
                                       conv_filters=[
                                           [[7, 7], 16],  # 1
                                           [[5, 5], 16],  # 2
                                           [[4, 4], 16],  # 3
                                           [[3, 3], 16],  # 4
                                           [[2, 2], 16]]  # 5
                                       ))

    output_concat = tf.concat(outputs, 1)
    total_output_len = len(outputs) * n_classes
    output_expand = tf.reshape(output_concat, [-1, total_output_len])
    print(output_expand)
    # Output layer
    final_output_size = 1
    W_Fin = tf.get_variable(
        name='w-fin-%d'%model_index,
        shape=[total_output_len, final_output_size],
        initializer=tf.contrib.layers.xavier_initializer()
    )

    B_Fin = tf.Variable(tf.constant(0.1, shape=[final_output_size]), name="B-final-out")
    output = tf.nn.sigmoid(tf.matmul(output_expand, W_Fin) + B_Fin)
    print(output)

    return output

def text_clf_model_ver1(input, conv_filters,char_size, embedding_size, model_index, n_classes, is_train,keep_prob):
    # specification
    fully_connect_hidden_layer_size = 84


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
    for i,shape in enumerate(conv_filters):
        with tf.name_scope("conv1-%d"%model_index + str(i+1)):
            conv = tf.layers.conv2d(conv,shape[1],shape[0],padding="SAME")
            print(conv)
        if i % 2 == 0:
            with tf.name_scope("max_pool-%d"%model_index + str(i+1)):
                conv = tf.layers.max_pooling2d(conv,[2,2],[2,2],padding="SAME")
                conv = tf.nn.relu(conv)
                print(conv)

        else:
            with tf.name_scope("conv2-%d"%model_index + str(i+1)):
                conv2 = tf.layers.conv2d(conv, shape[1], shape[0], padding="SAME")
                conv2 = tf.layers.batch_normalization(conv2, training=is_train)
                conv2 = tf.nn.relu(conv2)
                print(conv2)
            with tf.name_scope("residual-%d"%model_index + str(i + 1)):
                conv = conv + conv2

    with tf.name_scope("final_hidden%d"%model_index):
        lh = tf.contrib.layers.flatten(conv)
        lh = tf.layers.dense(lh,fully_connect_hidden_layer_size,activation=tf.nn.relu)
        lh = tf.nn.dropout(lh, keep_prob=keep_prob)
        print(lh)

    with tf.name_scope("final_layer%d"%model_index):
        w4 = tf.Variable(tf.random_normal([fully_connect_hidden_layer_size, n_classes], stddev=0.01))
        model = tf.nn.relu(tf.matmul(lh, w4))
        print(model)

    return model

def text_clf_model_ver2(input1, input2, char_size, embedding_size, is_train, keep_prob):
    # Create weights and biases
    # Embedding
    expand_input1,expand_input2 = text2embedding4CNN(input1,input2,char_size,embedding_size)

    # stem
    model1 = tf.layers.conv2d(expand_input1,filters=32,kernel_size=[7,7],strides=[1,1],padding="SAME",activation=tf.nn.relu)
    model2 = tf.layers.conv2d(expand_input2, filters=32, kernel_size=[7,7], strides=[1, 1], padding="SAME",activation=tf.nn.relu)

    #model1_temp = model1
    #model2_temp = model2

    model1b1 = tf.layers.max_pooling2d(model1,pool_size=[5,5],strides=[2,2],padding="SAME")
    model1b2 = tf.layers.conv2d(model1,filters=32,kernel_size=[5,5],strides=[2,2],padding="SAME",activation=tf.nn.relu)
    model2b1 = tf.layers.max_pooling2d(model2, pool_size=[5, 5], strides=[2, 2], padding="SAME")
    model2b2 = tf.layers.conv2d(model2, filters=32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                activation=tf.nn.relu)
    model1 = tf.concat([model1b1,model1b2],axis=3)
    model2 = tf.concat([model2b1,model2b2],axis=3)

    model1b1 = tf.layers.conv2d(model1,filters=32,kernel_size=[4,4],strides=[1,1],padding="SAME",activation=tf.nn.relu)
    model1b2 = tf.layers.conv2d(model1,filters=64,kernel_size=[2,2],strides=[1,1],padding="SAME",activation=tf.nn.relu)
    model1b2 = tf.layers.conv2d(model1b2,filters=32,kernel_size=[4,4],strides=[1,1],padding="SAME",activation=tf.nn.relu)

    model2b1 = tf.layers.conv2d(model2, filters=32, kernel_size=[4, 4], strides=[1, 1], padding="SAME",
                                activation=tf.nn.relu)
    model2b2 = tf.layers.conv2d(model2, filters=64, kernel_size=[2, 2], strides=[1, 1], padding="SAME",
                                activation=tf.nn.relu)
    model2b2 = tf.layers.conv2d(model2b2, filters=32, kernel_size=[4, 4], strides=[1, 1], padding="SAME",
                                activation=tf.nn.relu)
    model1 = tf.concat([model1b1,model1b2],axis=3)
    model2 = tf.concat([model2b1,model2b2],axis=3)
    summary_layer(model1,model2)
    model0 = model1 + model2
    summary_layer(model0)

    def layer1(model):
        model1 = tf.layers.average_pooling2d(model,pool_size=[3,3],strides=[2,2],padding="SAME")
        model1 = tf.layers.conv2d(model1,filters=32,kernel_size=[3,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
        model1 = tf.layers.conv2d(model1,filters=32,kernel_size=[1,3],strides=[1,1],padding="SAME",activation=tf.nn.relu)
        model2 = tf.layers.conv2d(model,filters=32,kernel_size=[3,3],strides=[2,2],padding="SAME",activation=tf.nn.relu)
        model = tf.concat([model1,model2],axis=3)
        return model

    model0 = layer1(model0)
    model1 = layer1(model1)
    model2 = layer1(model2)
    summary_layer(model1,model2)
    summary_layer(model0)

    model = tf.concat([model0,model1,model2],axis=3)
    summary_layer(model)
    def layer2(model):
        model1 = tf.layers.average_pooling2d(model,pool_size=[2,2],strides=[1,1],padding="SAME")
        model1 = tf.layers.conv2d(model1,filters=32,kernel_size=[2,2],strides=[2,2],padding="SAME",activation=tf.nn.relu)
        model2 = tf.layers.conv2d(model,filters=32,kernel_size=[1,3],strides=[1,1],padding="SAME",activation=tf.nn.relu)
        model2 = tf.layers.conv2d(model2,filters=32,kernel_size=[3,1],strides=[1,1],padding="SAME",activation=tf.nn.relu)
        model2 = tf.layers.conv2d(model2,filters=32,kernel_size=[2,2],strides=[2,2],padding="SAME",activation=tf.nn.relu)
        model = tf.concat([model1,model2],axis=3)
        return model

    model = layer2(model)
    summary_layer(model)
    '''
    # conv-layers
    conv1,conv2 = conv_layer2(expand_input1,expand_input2,filter=[7,7],filter_size=32,index=0)
    conv1, conv2 = conv_layer2(conv1, conv2, filter=[5, 5], filter_size=32, index=1)
    conv1, conv2 = conv_layer2(conv1, conv2, filter=[4, 4], filter_size=48, index=2)
    conv = tf.concat([conv1,conv2],3)
    conv = conv_layer(conv, filter=[3, 3], filter_size=64, index=3)
    conv = conv_layer(conv, filter=[2, 2], filter_size=64, index=4)
    '''

    # fully-connect-hidden layer
    with tf.name_scope("fc-layer"):
        fc_hidden = tf.contrib.layers.flatten(model)
        #input_dim = fc_hidden.get_shape().as_list()[-1]
        model = tf.layers.dense(fc_hidden,1,activation=tf.nn.sigmoid)
        #model = tf.layers.dropout(model,keep_prob)
        #model = tf.layers.dense(model,1,activation=tf.nn.sigmoid)

    summary_layer(model)

    return model



