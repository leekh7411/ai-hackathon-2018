import tensorflow as tf
'''
ResNet Reference by 
https://github.com/wenxinxu/resnet-in-tensorflow/blob/master/resnet.py
'''
FLAGS = tf.app.flags.FLAGS
BN_EPSILON = 0.001
tf.app.flags.DEFINE_float('weight_decay', 0.0002, '''scale for l2 regularization''')

def summary_layer(layer1,layer2='None'):
    print('----------------------------------------------------------')
    print(layer1,"\n",layer2)
    print('----------------------------------------------------------')

def text_clf_model_ver2(input1, input2, char_size, embedding_size, is_train, keep_prob):
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

    ''' residual block start '''
    # Conv layer-1
    with tf.name_scope("conv1"):
        conv1_input1 = tf.layers.conv2d(expand_input1, 32, [3,3], padding="SAME")
        conv1_input2 = tf.layers.conv2d(expand_input2, 32, [3,3], padding="SAME")
        summary_layer(conv1_input1,conv1_input2)

    # Batch norm 1
    with tf.name_scope("drop1"):
        #batch1_input1 = batch_normalization_layer(conv1_input1,dimension=[32],conv_index=1,input_index=1)
        #batch1_input2 = batch_normalization_layer(conv1_input2,dimension=[32],conv_index=1,input_index=2)
        drop1_input1 = tf.layers.dropout(conv1_input1,rate=keep_prob)
        drop1_input2 = tf.layers.dropout(conv1_input2,rate=keep_prob)
        summary_layer(drop1_input1, drop1_input2)

    # Relu layer 1
    with tf.name_scope("relu1"):
        relu1_input1 = tf.nn.relu(drop1_input1)
        relu1_input2 = tf.nn.relu(drop1_input2)
        summary_layer(relu1_input1, relu1_input2)

    # Conv layer 2
    with tf.name_scope("conv2"):
        conv2_input1 = tf.layers.conv2d(relu1_input1, 32, [3,3], padding="SAME")
        conv2_input2 = tf.layers.conv2d(relu1_input2, 32, [3,3], padding="SAME")
        summary_layer(conv2_input1, conv2_input2)

    # Batch norm 2
    with tf.name_scope("drop2"):
        #batch2_input1 = batch_normalization_layer(conv2_input1,dimension=[32],conv_index=2,input_index=1)
        #batch2_input2 = batch_normalization_layer(conv2_input2,dimension=[32],conv_index=2,input_index=2)
        drop2_input1 = tf.layers.dropout(conv2_input1,rate=keep_prob)
        drop2_input2 = tf.layers.dropout(conv2_input2,rate=keep_prob)
        summary_layer(drop2_input1, drop2_input2)

    # Relu layer 2
    with tf.name_scope("relu2"):
        relu2_input1 = tf.nn.relu(drop2_input1)
        relu2_input2 = tf.nn.relu(drop2_input2)
        summary_layer(relu2_input1, relu2_input2)

    # Conv layer 3
    with tf.name_scope("conv3"):
        conv3_input1 = tf.layers.conv2d(relu2_input1, 32, [3,3], padding="SAME")
        conv3_input2 = tf.layers.conv2d(relu2_input2, 32, [3,3], padding="SAME")
        summary_layer(conv3_input1, conv3_input2)

    # Residual 1
    with tf.name_scope("residual_sum1"):
        resd1_input1 = conv1_input1 + conv1_input2 + conv3_input1
        resd1_input2 = conv1_input1 + conv1_input2 + conv3_input2
        summary_layer(resd1_input1, resd1_input2)
    ''' residual block end '''


    # Batch norm fin
    with tf.name_scope("drop3"):
        #batch_fin_input1 = batch_normalization_layer(resd1_input1,[32], conv_index=99, input_index=1)
        #batch_fin_input2 = batch_normalization_layer(resd1_input2,[32], conv_index=99, input_index=2)
        drop3_input1 = tf.layers.dropout(resd1_input1,rate=keep_prob)
        drop3_input2 = tf.layers.dropout(resd1_input2,rate=keep_prob)
        summary_layer(drop3_input1, drop3_input2)



    ''' concat residual block start '''

    # final residual sum
    with tf.name_scope("final_resd_sum"):
        resd_fin_inputs = drop3_input1 + drop3_input2
        summary_layer(resd_fin_inputs)

    # conv-f 1
    with tf.name_scope("conv1-fin"):
        conv1_fin = tf.layers.conv2d(resd_fin_inputs, 64, [3,3], strides=[2,2],padding="SAME")
        conv1_fin = tf.layers.dropout(conv1_fin,rate=keep_prob)
        summary_layer(conv1_fin)

    # batch-f 1
    with tf.name_scope("batch1-fin"):
        batch1_fin = batch_normalization_layer(conv1_fin, [64], conv_index=1, input_index=0)
        summary_layer(batch1_fin)

    # relu-f 1
    with tf.name_scope("relu1-fin"):
        relu1_fin = tf.nn.relu(batch1_fin)
        summary_layer(relu1_fin)

    # conv-f 2
    with tf.name_scope("conv2-fin"):
        conv2_fin = tf.layers.conv2d(relu1_fin, 64, [3,3], strides=[2,2],padding="SAME")
        conv2_fin = tf.layers.dropout(conv2_fin,rate=keep_prob)
        summary_layer(conv2_fin)

    # batch-f 2
    with tf.name_scope("batch2-fin"):
        batch2_fin = batch_normalization_layer(conv2_fin, [64], conv_index=2, input_index=0)
        summary_layer(batch2_fin)

    # relu-f 2
    with tf.name_scope("relu2-fin"):
        relu2_fin = tf.nn.relu(batch2_fin)
        summary_layer(relu2_fin)

    # conv-f 3
    with tf.name_scope("conv3-fin"):
        conv3_fin = tf.layers.conv2d(relu2_fin, 64, [3,3], strides=[2,2],padding="SAME")
        conv3_fin = tf.layers.dropout(conv3_fin,rate=keep_prob)
        summary_layer(conv3_fin)

    # residual-f 1
    with tf.name_scope("resd1-fin"):
        input_channel = conv1_fin.get_shape().as_list()[-1]
        conv1_fin = tf.nn.avg_pool(conv1_fin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv1_fin = tf.nn.avg_pool(conv1_fin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #conv1_fin = tf.pad(conv1_fin, [[0, 0], [0, 0], [0, 0], [input_channel // 2, input_channel // 2]])
        resd1_fin = conv3_fin + conv1_fin
        summary_layer(resd1_fin)

    # batch-f 3
    with tf.name_scope("batch3-fin"):
        batch3_fin = batch_normalization_layer(resd1_fin, [64], conv_index=3, input_index=0)
        summary_layer(batch3_fin)

    # relu-f 3
    with tf.name_scope("relu3-fin"):
        relu3_fin = tf.nn.relu(batch3_fin)
        summary_layer(relu3_fin)

    # conv-f 4
    with tf.name_scope("conv4-fin"):
        conv4_fin = tf.layers.conv2d(relu3_fin, 64, [2,2],strides=[2,2], padding="SAME")
        conv4_fin = tf.layers.dropout(conv4_fin,rate=keep_prob)
        summary_layer(conv4_fin)

    # batch-f 4
    with tf.name_scope("batch4-fin"):
        batch4_fin = batch_normalization_layer(conv4_fin,[64],conv_index=4,input_index=0)
        summary_layer(batch4_fin)

    # relu-f 4
    with tf.name_scope("relu4-fin"):
        relu4_fin = tf.nn.relu(batch4_fin)
        summary_layer(relu4_fin)

    # conv-f 5
    with tf.name_scope("conv5-fin"):
        conv5_fin = tf.layers.conv2d(relu4_fin, 64, [2, 2], strides=[2, 2], padding="SAME")
        conv5_fin = tf.layers.dropout(conv5_fin,rate=keep_prob)
        summary_layer(conv5_fin)

    # batch-f 5
    with tf.name_scope("batch5-fin"):
        batch5_fin = batch_normalization_layer(conv5_fin, [64], conv_index=5, input_index=0)
        summary_layer(batch5_fin)

    # relu-f 5
    with tf.name_scope("relu5-fin"):
        relu5_fin = tf.nn.relu(batch5_fin)
        summary_layer(relu5_fin)

    # fully-connect-hidden layer
    with tf.name_scope("fc-layer"):
        fc_hidden = tf.contrib.layers.flatten(relu5_fin)
        fc_hidden = tf.layers.dense(fc_hidden, 2048, activation=tf.nn.relu)
        fc_hidden = tf.layers.dropout(fc_hidden,rate=keep_prob)
        fc_hidden = tf.layers.dense(fc_hidden, 256, activation=tf.nn.relu)
        fc_hidden = tf.layers.dropout(fc_hidden, rate=keep_prob)
        fc_hidden = tf.layers.dense(fc_hidden, 128, activation=tf.nn.relu)
        fc_hidden = tf.layers.dropout(fc_hidden, rate=keep_prob)
        fc_hidden = tf.layers.dense(fc_hidden, 32, activation=tf.nn.relu)
        fc_hidden = tf.layers.dropout(fc_hidden, rate=keep_prob)
        fc_hidden = tf.layers.dense(fc_hidden, 1, activation=tf.nn.sigmoid)
        summary_layer(fc_hidden)

    model = fc_hidden

    return model



def text_clf_model_ver1(_x, _character_size, _embedding_size, model_index, n_classes, is_train):
    # specification

    conv_filters = [
        [[7,7],64],#0
        [[6,6],64],#1
        [[5,5],64],#2
        [[4,4],64],#3
        [[3,3],64],#4
        [[2,2],64]]#5
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
    conv = embedded_expand
    for i,shape in enumerate(conv_filters):
        with tf.name_scope("conv1-" + str(i+1)):
            conv = tf.layers.conv2d(conv,shape[1],shape[0],padding="SAME")
            print(conv)
        if i % 2 == 0:
            with tf.name_scope("avg_pool-" + str(i+1)):
                conv = tf.layers.max_pooling2d(conv,[2,2],[2,2],padding="SAME")
                print(conv)

        else:
            with tf.name_scope("conv2-" + str(i+1)):
                conv2 = tf.layers.conv2d(conv,shape[1],shape[0],padding="SAME")
                conv2 = tf.layers.batch_normalization(conv2, training=is_train)
                conv2 = tf.nn.relu(conv2)
                print(conv2)
            with tf.name_scope("residual-" + str(i + 1)):
                conv = conv + conv2


        '''with tf.name_scope("drop_out-" + str(i+1)):
            conv = tf.nn.dropout(conv,keep_prob=keep_prob)
        '''

    with tf.name_scope("final_hidden"):
        lh = tf.contrib.layers.flatten(conv)
        lh = tf.layers.dense(lh,fully_connect_hidden_layer_size,activation=tf.nn.relu)
        #lh = tf.nn.dropout(lh, keep_prob=keep_prob)
        print(lh)

    with tf.name_scope("final_layer"):
        w4 = tf.Variable(tf.random_normal([fully_connect_hidden_layer_size, n_classes], stddev=0.01))
        model = tf.nn.relu(tf.matmul(lh, w4))
        print(model)

    return model

def batch_normalization_layer(input_layer, dimension, conv_index, input_index):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''

    with tf.name_scope("batch_norm-"+ str(conv_index) + str(input_index)):
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta-'+ str(conv_index) + str(input_index), dimension, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma-'+ str(conv_index) + str(input_index), dimension, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''

    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h




def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output

def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer

def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output

def inference(input_tensor_batch, n, reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 16)
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32)
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 10)
        layers.append(output)

    return layers[-1]




def text_clf_model_ver4(input1,input2,char_size,embedding_size,phase):
    def dense(x, size, scope):
        return tf.layers.dense(x,size,activation=None,name=scope)

    def dense_batch_relu(x, n_hidden, phase, scope):
        with tf.variable_scope(scope):
            h1 = dense(x,n_hidden,scope=scope)
            h2 = tf.layers.batch_normalization(h1,center=True,scale=True,training=phase,name=scope)
            return tf.nn.relu(h2)

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

        expand_input1 = tf.contrib.layers.flatten(expand_input1)
        expand_input2 = tf.contrib.layers.flatten(expand_input2)

        summary_layer(expand_input1, expand_input2)

    h1_i1 = dense_batch_relu(expand_input1,256,phase,'layer1-i1')
    h1_i2 = dense_batch_relu(expand_input2,256,phase,'layer1-i2')
    summary_layer(h1_i1,h1_i2)

    h2_i1 = dense_batch_relu(h1_i1,128,phase,'layer2-i1')
    h2_i2 = dense_batch_relu(h1_i2,128,phase,'layer2-i2')
    summary_layer(h2_i1,h2_i2)

    h3 = h2_i1 + h2_i2
    h3 = dense_batch_relu(h3,64,phase,'layer3-all')
    summary_layer(h3)

    logits = dense(h3,1,'logits')
    summary_layer(logits)

    return tf.nn.sigmoid(logits)


def text_clf_model_ver3(input1, input2, char_size, embedding_size, is_train):
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


    # Conv layer-1
    conv1_i1 = expand_input1
    conv1_i2 = expand_input2
    conv1_input1 = expand_input1
    conv1_input2 = expand_input2
    for i in range(3):
        # conv
        conv1_input1 = tf.layers.conv2d(conv1_input1, 64, [3, 3], padding="SAME")
        conv1_input2 = tf.layers.conv2d(conv1_input2, 64, [3, 3], padding="SAME")
        conv1_b1 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=is_train)
        conv1_b2 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=is_train)
        conv1_input1 = tf.layers.batch_normalization(conv1_input1, training=is_train)
        conv1_input2 = tf.layers.batch_normalization(conv1_input2, training=is_train)
        conv1_input1 = tf.nn.relu(tf.nn.bias_add(conv1_input1,conv1_b1))
        conv1_input2 = tf.nn.relu(tf.nn.bias_add(conv1_input2,conv1_b2))
        conv1_input1 = tf.layers.max_pooling2d(conv1_input1,[2,2],[2,2],padding="SAME")
        conv1_input2 = tf.layers.max_pooling2d(conv1_input2,[2,2],[2,2],padding="SAME")
        summary_layer(conv1_input1, conv1_input2)

    # Residual 1
    with tf.name_scope("resd1"):
        resd1_inputs = conv1_input1 + conv1_input2
        summary_layer(resd1_inputs)

    # Conv layer 2
    with tf.name_scope("conv2"):
        conv2_inputs = tf.layers.conv2d(resd1_inputs, 64, [2,2], padding="SAME")
        conv2_inputs = tf.layers.max_pooling2d(conv2_inputs,[2,2],[2,2],padding="SAME")
        summary_layer(conv2_inputs)

    # Batch norm 2
    with tf.name_scope("batch2"):
        batch2_inputs = tf.layers.batch_normalization(conv2_inputs, center=True, scale=True, training=is_train)
        summary_layer(batch2_inputs)

    # Relu layer 2
    with tf.name_scope("relu2"):
        relu2_inputs = tf.nn.relu(batch2_inputs)
        summary_layer(relu2_inputs)

    # FC-Net
    with tf.name_scope("FC-Net"):
        fc_hidden = tf.contrib.layers.flatten(relu2_inputs)
        fc_hidden = tf.layers.dense(fc_hidden, 2048, activation=None)
        fc_hidden = tf.layers.batch_normalization(fc_hidden,center=True,scale=True,training=is_train)
        fc_hidden = tf.nn.relu(fc_hidden)
        summary_layer(fc_hidden)

        fc_w = create_variables(name='fc_weights', shape=[256, 1], is_fc_layer=True,
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        fc_b = create_variables(name='fc_bias', shape=[1], initializer=tf.zeros_initializer())
        fc_h = tf.matmul(fc_hidden, fc_w) + fc_b
        fc_h = tf.layers.batch_normalization(fc_h,center=True,scale=True,training=is_train)
        summary_layer(fc_h)

    # For binary clf.. sigmoid
    model = tf.nn.sigmoid(fc_h)
    summary_layer(model)
    return model