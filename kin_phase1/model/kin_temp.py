"""

useless codes but 'may be..'

"""

def fin__layer(input,keep_prob):
    output = tf.layers.average_pooling2d(input,pool_size=[3,3],strides=[1,1],padding="SAME")
    output = tf.layers.dropout(output,rate=keep_prob)
    output = tf.layers.flatten(output)
    output = tf.layers.dense(output,1,activation=tf.nn.sigmoid,use_bias=True)
    return output

def inception_V4(input,char_size, embedding_size,keep_prob):
    # CharVector 2 Embedding
    model = text2embedding4CNN1(input,char_size,embedding_size)
    model = stem_module(model)

    # 4 x inception-A
    for i in range(4):
        model = inception_A(model)

    # Reduction A
    model = reduction_A(model)

    # 7 x inception-B
    for i in range(7):
        model = inception_B(model)

    # Reduction B
    model = reduction_B(model)

    # 3 x inception-C
    for i in range(3):
        model = inception_C(model)

    model = fin__layer(model,keep_prob)

    return model
def stem_module(input):
    input = tf.layers.conv2d(input, filters=32, kernel_size=[3, 3], strides=[2, 2], padding="SAME")
    input = tf.layers.conv2d(input, filters=32, kernel_size=[3, 3], strides=[1, 1], padding="SAME")
    input = tf.layers.conv2d(input, filters=64, kernel_size=[3, 3], strides=[1, 1], padding="SAME")
    summary_layer(input,tag='stem-first')

    max_pool = tf.layers.max_pooling2d(input,[3,3],strides=[2,2],padding="SAME")
    conv = tf.layers.conv2d(input,filters=32,kernel_size=[3,3],strides=[2,2],padding="SAME")
    input = tf.concat([max_pool,conv],axis=3)
    summary_layer(input,tag='stem-after-bridge-concat-second')

    bridge1 = tf.layers.conv2d(input,filters=64,kernel_size=[1,1],padding="SAME")
    bridge1 = tf.layers.conv2d(bridge1,filters=96,kernel_size=[3,3],strides=[1,1],padding="SAME")
    bridge2 = tf.layers.conv2d(input,filters=64,kernel_size=[1,1],padding="SAME")
    bridge2 = tf.layers.conv2d(bridge2,filters=64,kernel_size=[7,1],strides=[1,1],padding="SAME")
    bridge2 = tf.layers.conv2d(bridge2,filters=64,kernel_size=[1,7],strides=[1,1],padding="SAME")
    bridge2 = tf.layers.conv2d(bridge2,filters=96,kernel_size=[3,3],strides=[1,1],padding="SAME")
    input = tf.concat([bridge1,bridge2],axis=3)
    summary_layer(input,tag='stem-after-bridge-concat-third')

    bridge1 = tf.layers.max_pooling2d(input,pool_size=[3,3],strides=[1,1],padding="SAME")
    bridge2 = tf.layers.conv2d(input,filters=192,kernel_size=[3,3],strides=[1,1],padding="SAME")
    input = tf.concat([bridge1,bridge2],axis=3)
    summary_layer(input,tag='stem-concat-last')
    return input

def inception_A(input):
    bridge1 = tf.layers.average_pooling2d(input,pool_size=[3,3],strides=[1,1],padding="SAME")
    bridge1 = tf.layers.conv2d(bridge1,filters=96,kernel_size=[1,1],strides=[1,1],padding="SAME")
    summary_layer(bridge1,tag='incep-A-(1)-avg_pool-[1x1]Conv-96')
    bridge2 = tf.layers.conv2d(input,filters=96,kernel_size=[1,1],strides=[1,1],padding="SAME")
    summary_layer(bridge2,tag='incep-A-(2)-[1x1]Conv-96')
    bridge3 = tf.layers.conv2d(input,filters=64,kernel_size=[1,1],strides=[1,1],padding="SAME")
    bridge3 = tf.layers.conv2d(bridge3,filters=96,kernel_size=[3,3],strides=[1,1],padding="SAME")
    summary_layer(bridge3,tag='incep-A-(3)-[1x1]Conv-64-[3x3]Conv-96')
    bridge4 = tf.layers.conv2d(input,filters=64,kernel_size=[1,1],strides=[1,1],padding="SAME")
    bridge4 = tf.layers.conv2d(bridge4,filters=96,kernel_size=[3,3],strides=[1,1],padding="SAME")
    bridge4 = tf.layers.conv2d(bridge4, filters=96, kernel_size=[3, 3], strides=[1, 1], padding="SAME")
    summary_layer(bridge4,tag='incep-A-(4)-[1x1]Conv-64-[3x3]Conv-96-[3x3]Conv-96')
    output = tf.concat([bridge1,bridge2,bridge3,bridge4],axis=3)
    summary_layer(output,tag='incep-A-last')
    return output

def reduction_A(input):
    bridge1 = tf.layers.max_pooling2d(input,pool_size=[3,3],strides=[2,2],padding="SAME")
    bridge2 = tf.layers.conv2d(input,filters=384,kernel_size=[3,3],strides=[2,2],padding="SAME")
    bridge3 = tf.layers.conv2d(input,filters=192,kernel_size=[1,1],strides=[1,1],padding="SAME")
    bridge3 = tf.layers.conv2d(bridge3,filters=224,kernel_size=[3,3],strides=[1,1],padding="SAME")
    bridge3 = tf.layers.conv2d(bridge3,filters=256,kernel_size=[3,3],strides=[2,2],padding="SAME")
    summary_layer(bridge1,tag="reduction-A-[3,3][2,2]max-pool")
    summary_layer(bridge2,tag="reduction-A-[3,3][2,2]Conv-384")
    summary_layer(bridge3,tag="reduction-A-[3,3][1,1]Conv-224-[3,3][2,2]Conv-256")
    output = tf.concat([bridge1,bridge2,bridge3],axis=3)
    summary_layer(output,tag="reduction-A-last")
    return output

def inception_B(input):
    bridge1 = tf.layers.average_pooling2d(input,pool_size=[3,3],strides=[1,1],padding="SAME")
    bridge1 = tf.layers.conv2d(bridge1,filters=128,kernel_size=[1,1],strides=[1,1],padding="SAME")
    summary_layer(bridge1, tag="inception-B-[1,1][1,1]Avg_pool-[1,1][1,1]Conv-128")
    bridge2 = tf.layers.conv2d(input,filters=384,kernel_size=[1,1],strides=[1,1],padding="SAME")
    summary_layer(bridge2, tag="inception-B-[1,1][1,1]Conv-384")
    bridge3 = tf.layers.conv2d(input,filters=192,kernel_size=[1,1],strides=[1,1],padding="SAME")
    bridge3 = tf.layers.conv2d(bridge3,filters=224,kernel_size=[1,7],strides=[1,1],padding="SAME")
    bridge3 = tf.layers.conv2d(bridge3,filters=256,kernel_size=[7,1],strides=[1,1],padding="SAME")
    summary_layer(bridge3, tag="inception-B-[1,1][1,1]Conv-192-[1,7][1,1]Conv-224-[7,1][1,1]Conv-256")
    bridge4 = tf.layers.conv2d(input,filters=192,kernel_size=[1,1],strides=[1,1],padding="SAME")
    bridge4 = tf.layers.conv2d(bridge4,filters=192,kernel_size=[1,7],strides=[1,1],padding="SAME")
    bridge4 = tf.layers.conv2d(bridge4,filters=224,kernel_size=[7,1],strides=[1,1],padding="SAME")
    bridge4 = tf.layers.conv2d(bridge4, filters=224, kernel_size=[1, 7], strides=[1, 1], padding="SAME")
    bridge4 = tf.layers.conv2d(bridge4, filters=256, kernel_size=[7, 1], strides=[1, 1], padding="SAME")
    summary_layer(bridge4, tag="inception-B-[1,1][1,1]Conv192-[1,7][1,1]Conv192-[7,1][1,1]Conv224-[1,7][1,1]Conv224-[7,1][1,1]Conv256")
    output = tf.concat([bridge1,bridge2,bridge3,bridge4],axis=3)
    summary_layer(output,tag="inception-B-last")
    return output

def reduction_B(input):
    bridge1 = tf.layers.max_pooling2d(input,pool_size=[3,3],strides=[2,2],padding="SAME")
    summary_layer(bridge1,tag="reduction-B-[3,3][2,2]max-pool")
    bridge2 = tf.layers.conv2d(input,filters=192,kernel_size=[1,1],strides=[1,1],padding="SAME")
    bridge2 = tf.layers.conv2d(bridge2,filters=192,kernel_size=[3,3],strides=[2,2],padding="SAME")
    summary_layer(bridge2,tag="reduction-B[1,1][1,1]Conv-192-[3,3][2,2]Conv-192")
    bridge3 = tf.layers.conv2d(input,filters=256,kernel_size=[1,1],strides=[1,1],padding="SAME")
    bridge3 = tf.layers.conv2d(bridge3,filters=256,kernel_size=[1,7],strides=[1,1],padding="SAME")
    bridge3 = tf.layers.conv2d(bridge3, filters=320, kernel_size=[7, 1], strides=[1, 1], padding="SAME")
    bridge3 = tf.layers.conv2d(bridge3, filters=320, kernel_size=[3, 3], strides=[2, 2], padding="SAME")
    summary_layer(bridge3, tag="reduction-B[1,1][1,1]Conv-256-[1,7][1,1]Conv-256-[7,1][1,1]Conv-320-[3,3][2,2]Conv-320")
    output = tf.concat([bridge1,bridge2,bridge3],axis=3)
    summary_layer(output,tag="reduction-B-last")
    return output

def inception_C(input):
    bridge1 = tf.layers.average_pooling2d(input,pool_size=[3,3],strides=[1,1],padding="SAME")
    bridge1 = tf.layers.conv2d(bridge1,filters=256,kernel_size=[1,1],strides=[1,1],padding="SAME")
    summary_layer(bridge1,tag="inception-C-[1,1][1,1]avg_pool-[1,1][1,1]Conv-256")
    bridge2 = tf.layers.conv2d(input,filters=256,kernel_size=[1,1],strides=[1,1],padding="SAME")
    summary_layer(bridge2,tag="inception-C-[1,1][1,1]Conv-256")
    bridge3 = tf.layers.conv2d(input,filters=384,kernel_size=[1,1],strides=[1,1],padding="SAME")
    bridge3b1 = tf.layers.conv2d(bridge3,filters=256,kernel_size=[1,3],strides=[1,1],padding="SAME")
    bridge3b2 = tf.layers.conv2d(bridge3,filters=256,kernel_size=[3,1],strides=[1,1],padding="SAME")
    summary_layer(bridge3b1,bridge3b2,tag="inception-C-[1,1]Conv384-(1)-[1,3]Conv256-(2)-[3,1]Conv256")
    bridge4 = tf.layers.conv2d(input,filters=384,kernel_size=[1,1],strides=[1,1],padding="SAME")
    bridge4 = tf.layers.conv2d(bridge4,filters=448,kernel_size=[1,3],strides=[1,1],padding="SAME")
    bridge4 = tf.layers.conv2d(bridge4,filters=512,kernel_size=[3,1],strides=[1,1],padding="SAME")
    bridge4b1 = tf.layers.conv2d(bridge4,filters=256,kernel_size=[3,1],strides=[1,1],padding="SAME")
    bridge4b2 = tf.layers.conv2d(bridge4,filters=256,kernel_size=[1,3],strides=[1,1],padding="SAME")
    summary_layer(bridge4b1,bridge4b2,tag="inception-C-[1,1]Conv384-[1,3]Conv448-[3,1]Conv512-(1)-[3,1]Conv256-(2)-[1,3]Conv256")
    output = tf.concat([bridge1,bridge2,bridge3b1,bridge3b2,bridge4b1,bridge4b2],axis=3)
    summary_layer(output,tag="inception-C-last")
    return output

def fin__layer(input,keep_prob):
    output = tf.layers.average_pooling2d(input,pool_size=[3,3],strides=[1,1],padding="SAME")
    output = tf.layers.dropout(output,rate=keep_prob)
    output = tf.layers.flatten(output)
    output = tf.layers.dense(output,1,activation=tf.nn.sigmoid,use_bias=True)
    return output

###################################################################################################
def inception_module(input):
    out1 = tf.layers.conv2d(input,[1,1],padding="SAME",kernel_size=32)
    summary_layer(out1)
    out2 = tf.layers.conv2d(input,[1,1],padding="SAME",kernel_size=32)
    out2 = tf.layers.conv2d(out2,[3,3],padding="SAME",kernel_size=32)
    summary_layer(out2)
    out3 = tf.layers.conv2d(input,[1,1],padding="SAME",kernel_size=32)
    out3 = tf.layers.conv2d(out3,[5,5],padding="SAME",kernel_size=32)
    summary_layer(out3)
    out4 = tf.layers.max_pooling2d(input,[3,3],[1,1],padding="SAME")
    summary_layer(out4)
    out = tf.concat([out1,out2,out3,out4],axis=3)
    summary_layer(out)
    return out

def text_clf_model_ver6(input1,input2,char_size,embedding_size,keep_prob,is_train):

    # Text Embedding
    input1, input2 = text2embedding4CNN(input1,input2,char_size,embedding_size)

    # For Ensemble learning method
    outputs = []


def text_clf_model_ver3(input1, input2, char_size, embedding_size, is_train, keep_prob):
    # Text Embedding
    input1, input2 = text2embedding4CNN(input1, input2, char_size, embedding_size)
    # Conv layer-1
    conv1_input1 = input1
    conv1_input2 = input2
    for i in range(3):
        # conv
        conv1_input1_temp = conv1_input1
        conv1_input2_temp = conv1_input2
        conv1_input1 = tf.layers.conv2d(conv1_input1, 32, [3, 3], padding="SAME")
        conv1_input2 = tf.layers.conv2d(conv1_input2, 32, [3, 3], padding="SAME")
        conv1_b1 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=is_train)
        conv1_b2 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=is_train)
        conv1_input1 = tf.layers.batch_normalization(conv1_input1, training=is_train)
        conv1_input2 = tf.layers.batch_normalization(conv1_input2, training=is_train)
        conv1_input1 = tf.nn.relu(tf.nn.bias_add(conv1_input1,conv1_b1))
        conv1_input2 = tf.nn.relu(tf.nn.bias_add(conv1_input2,conv1_b2))
        conv1_input1 = tf.layers.conv2d(conv1_input1, 32, [3, 3], padding="SAME")
        conv1_input2 = tf.layers.conv2d(conv1_input2, 32, [3, 3], padding="SAME")
        conv1_b1 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=is_train)
        conv1_b2 = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=is_train)
        conv1_input1 = tf.layers.batch_normalization(conv1_input1, training=is_train)
        conv1_input2 = tf.layers.batch_normalization(conv1_input2, training=is_train)
        conv1_input1 += conv1_input1_temp
        conv1_input2 += conv1_input2_temp
        conv1_input1 = tf.nn.relu(tf.nn.bias_add(conv1_input1, conv1_b1))
        conv1_input2 = tf.nn.relu(tf.nn.bias_add(conv1_input2, conv1_b2))
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
        fc_hidden = tf.layers.dense(fc_hidden, 256, activation=None)
        fc_hidden = tf.layers.batch_normalization(fc_hidden, center=True, scale=True, training=is_train)
        fc_hidden = tf.nn.relu(fc_hidden)
        summary_layer(fc_hidden)

        fc_w = create_variables(name='fc_weights', shape=[256, 1], is_fc_layer=True,
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        fc_b = create_variables(name='fc_bias', shape=[1], initializer=tf.zeros_initializer())
        fc_h = tf.matmul(fc_hidden, fc_w) + fc_b
        fc_h = tf.layers.batch_normalization(fc_h, center=True, scale=True, training=is_train)
        summary_layer(fc_h)

    # For binary clf.. sigmoid
    model = tf.nn.sigmoid(fc_h)
    summary_layer(model)
    return model

#?#####################################################################################################################

def text_clf_model_ver5(input,is_train,keep_prob):
    print(input)
    # Reference by VGG16
    # tensorflow : https://github.com/abhaydoke09/Bilinear-CNN-TensorFlow/blob/master/core/bcnn_DD_woft.py
    # conv1_1
    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32, # channel size 1
                                                 stddev=1e-1), trainable=False, name='weights')
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=False, name='biases')
        conv1_1 = tf.nn.bias_add(conv, biases)

    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False, name='weights')
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False, name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)

    # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False, name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope)

    # pool2
    pool2 = tf.nn.max_pool(conv2_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False, name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)

    # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False,  name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                               trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)

    # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False, name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope)

    # pool3
    pool3 = tf.nn.max_pool(conv3_3,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool3')

    # conv4_1
    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False, name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope)

    # conv4_2
    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False,   name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                              trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope)

    # conv4_3
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False, name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope)

    # pool4
    pool4 = tf.nn.max_pool(conv4_3,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool4')

    # conv5_1
    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False, name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope)


    # conv5_2
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False,  name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                              trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope)

    # conv5_3
    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), trainable=False, name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=False, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope)

    phi_I = tf.einsum('ijkm,ijkn->imn', conv5_3, conv5_3)

    phi_I = tf.reshape(phi_I, [-1, 64 * 64])
    print('Shape of phi_I after reshape', phi_I.get_shape())

    #phi_I = tf.divide(phi_I, 784.0)
    #print('Shape of phi_I after division', phi_I.get_shape())

    #y_ssqrt = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))
    #print('Shape of y_ssqrt', y_ssqrt.get_shape())

    z_l2 = tf.nn.l2_normalize(phi_I, dim=1)
    print('Shape of z_l2', z_l2.get_shape())

    with tf.name_scope('fc-new') as scope:
        fw1 = tf.get_variable('weights1', [64 * 64, 512], initializer=tf.contrib.layers.xavier_initializer(),
                               trainable=True)
        fb1 = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32), name='biases1', trainable=True)
        fl1 = tf.nn.bias_add(tf.matmul(z_l2, fw1), fb1)
        fl1 = tf.nn.dropout(fl1,keep_prob=keep_prob)
        fl1 = tf.nn.relu(fl1)

        fw2 = tf.get_variable('weights2', [512, 1], initializer=tf.contrib.layers.xavier_initializer(),
                              trainable=True)
        fb2 = tf.Variable(tf.constant(1.0, shape=[1], dtype=tf.float32), name='biases2', trainable=True)
        fl2 = tf.nn.bias_add(tf.matmul(fl1, fw2), fb2)
        '''fl2 = tf.nn.dropout(fl2, keep_prob=keep_prob)
        fl2 = tf.nn.relu(fl2)

        fw3 = tf.get_variable('weights3', [256, 32], initializer=tf.contrib.layers.xavier_initializer(),
                              trainable=True)
        fb3 = tf.Variable(tf.constant(1.0, shape=[32], dtype=tf.float32), name='biases3', trainable=True)
        fl3 = tf.nn.bias_add(tf.matmul(fl2, fw3), fb3)
        fl3 = tf.nn.dropout(fl3, keep_prob=keep_prob)
        fl3 = tf.nn.relu(fl3)

        fw4 = tf.get_variable('weights4', [32, 1], initializer=tf.contrib.layers.xavier_initializer(),
                              trainable=True)
        fb4 = tf.Variable(tf.constant(1.0, shape=[1], dtype=tf.float32), name='biases4', trainable=True)
        fl4 = tf.nn.bias_add(tf.matmul(fl3, fw4), fb4)'''

    model = tf.nn.sigmoid(fl2)


    return model

#?#####################################################################################################################




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







'''
dataset = KinQueryDataset(LOCAL_DATASET_PATH,400)

for i, (data1,data2, labels) in enumerate(_batch_loader(dataset, 100)):
    for i in range(len(labels)):
        print(data1[i])
        print(data2[i])
        print(labels[i])

corpus_raw =  "지식인 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다."
corpus_raw = corpus_raw.replace('.',' ')
words = []
for word in corpus_raw.split():
    #print(word)
    words.append(word)

words = set(words) # remove duplicate word

# dictionary
word2int = {}
int2word = {}

vocab_size = len(words) # gives the total number of unique words
embedding_size = 50
for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

idx = word2int['지식인']
print(int2word[idx])

#---------------------------------------------------------------
# raw sentences is a list of sentences.
raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())

print(sentences)

data = []
WINDOW_SIZE = 2
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([word, nb_word])

print(data)

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size,dtype=float)
    temp[data_point_index] = 1.0
    return temp

x_train = [] # input word
y_train = [] # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size) + to_one_hot(word2int[ data_word[1] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))
# convert them to numpy arrays
x_train = np.asarray(x_train)
x_train = x_train.flatten()
y_train = np.asarray(y_train)

print(x_train)


embeddings = tf.Variable(
    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

nce_weights = tf.Variable(
  tf.truncated_normal([vocab_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

embed = tf.nn.embedding_lookup(embeddings, dataset.queries1)

'''""""""


"""
Copyright 2018 NAVER Corp.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os
import numpy as np
import tensorflow as tf
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from kin_dataset import KinQueryDataset, preprocess
from kin_cnn_models import text_clf_model_ver3, text_clf_model_ver5
from nsml import GPU_NUM
# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)


        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output, feed_dict={
                                input: preprocessed_data,
                                is_training: False,
                                keep_prob:1})

        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

def data_normalization(data):
    if is_data_norm:
        return ((data - np.mean(data)) / np.std(data))
    else:
        return data

def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다
    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':
    #if GPU_NUM:
        args = argparse.ArgumentParser()
        # DONOTCHANGE: They are reserved for nsml
        args.add_argument('--mode', type=str, default='train')
        args.add_argument('--pause', type=int, default=0)
        args.add_argument('--iteration', type=str, default='0')

        # User options
        args.add_argument('--output', type=int, default=1)
        args.add_argument('--epochs', type=int, default=100)
        args.add_argument('--batch', type=int, default=2000)
        args.add_argument('--strmaxlen', type=int, default=25)
        args.add_argument('--w2v_size',type=int, default=50)
        args.add_argument('--embedding', type=int, default=8)
        args.add_argument('--threshold', type=float, default=0.5)
        args.add_argument('--lr',type=float,default=0.001)
        config = args.parse_args()

        if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
            DATASET_PATH = '/home/leekh7411/PycharmProject/ai-hackathon-2018/kin_phase1/sample_data/kin/'

        L1_INPUT = config.embedding * config.strmaxlen  # 8 x 400
        FIN_OUTPUT = 1
        learning_rate = config.lr
        learning_rate_tf = tf.placeholder(tf.float32,[],name="lr")
        train_decay = .99
        character_size = 400
        w2v_size = config.w2v_size
        drop_out_val = 0.8
        keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool)
        is_train = True
        is_data_norm = False
        beta = 0.01

        # Input & Output layer
        input = tf.placeholder(tf.float32, [None, config.strmaxlen * 2, config.w2v_size, 1], name="input-x")
        y_ = tf.placeholder(tf.float32, [None, FIN_OUTPUT], name="output-y")


        output  = text_clf_model_ver5(
            input = input,
            is_train=is_train,
            keep_prob=keep_prob
        )

        # loss와 optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            with tf.name_scope("loss-optimizer"):
                # Binary Cross Entropy
                binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(output,1e-10,1.0))) - (1-y_) * tf.log(tf.clip_by_value(1-output,1e-10,1.0)))

                # L1 Regularization + L2 Regularization
                #l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
                #weights = tf.trainable_variables()  # all vars of your graph
                #regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
                #binary_cross_entropy = tf.reduce_mean(binary_cross_entropy + regularization_penalty)

                # Adam Opt
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(binary_cross_entropy)
                #train_step = tf.train.MomentumOptimizer(learning_rate=0.9, momentum=0.9).minimize(binary_cross_entropy)


        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # ========================================================================================#

        # DONOTCHANGE: Reserved for nsml
        bind_model(sess=sess, config=config)


        # DONOTCHANGE: Reserved for nsml
        if config.pause:
            nsml.paused(scope=locals())

        if config.mode == 'train':
            # 데이터를 로드합니다.
            dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
            dataset_len = len(dataset)
            one_batch_size = dataset_len//config.batch
            if dataset_len % config.batch != 0:
                one_batch_size += 1
            # epoch마다 학습을 수행합니다.
            for epoch in range(config.epochs):
                avg_loss = 0.0
                avg_val_acc = 0.0
                avg_val_loss = 0.0

                for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                    # Divide Cross Validation Set
                    test_idx = (int)(len(labels) * 0.9)
                    train_data = data[:test_idx]

                    test_data = data[test_idx:]

                    train_labels = labels[:test_idx]
                    test_labels = labels[test_idx:]

                    #shuffle
                    s = np.random.permutation(train_labels.shape[0])
                    train_data = train_data[s]
                    train_labels = train_labels[s]

                    _, loss = sess.run([train_step, binary_cross_entropy],
                                       feed_dict={
                                           input: train_data,
                                           y_: train_labels,
                                           learning_rate_tf: learning_rate,
                                           is_training: True,
                                           keep_prob: drop_out_val
                                       })

                    # Test Validation Set
                    pred = sess.run(output, feed_dict={input: test_data,y_:test_labels ,is_training: False, keep_prob:1})
                    pred_clipped = np.array(pred > config.threshold, dtype=np.int)
                    is_correct = tf.equal(pred_clipped, test_labels)
                    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

                    # Get Validation Loss
                    val_acc = accuracy.eval(
                        feed_dict={input: test_data,
                                   y_: test_labels,
                                   is_training: False,
                                   keep_prob:1})

                    val_loss = sess.run(binary_cross_entropy,
                                        feed_dict={input: test_data,
                                                   y_:test_labels,
                                                   is_training: False,
                                                   keep_prob:1})

                    print('Batch : ', i + 1, '/', one_batch_size, ', Batch Size:', one_batch_size ,
                          ', BCE in this minibatch: ', float(loss),
                          " Valid loss :", float(val_loss),
                          " Valid score:", float(val_acc) * 100)
                    avg_loss += float((loss))
                    avg_val_acc += float((val_acc))
                    avg_val_loss += float(val_loss)

                print('epoch:', epoch, ' train_loss:', float(avg_loss / (one_batch_size)),' valid_loss:',float(avg_val_loss/(one_batch_size)) ,' valid_acc:',
                      float(avg_val_acc / (one_batch_size)) * 100)

                nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                            train__loss=float(avg_loss/one_batch_size), step=epoch, val_loss=float(avg_val_loss / (one_batch_size)))
                learning_rate = learning_rate * train_decay

                # DONOTCHANGE (You can decide how often you want to save the model)
                nsml.save(epoch)

        # 로컬 테스트 모드일때 사용합니다
        # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
        # [(0.3, 0), (0.7, 1), ... ]
        else:
            with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
                queries = f.readlines()
            res = []
            for batch in _batch_loader(queries, config.batch):
                temp_res = nsml.infer(batch)
                res += temp_res
            print(res)




#######################################################################################################################
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

def textRNNModel(input, n_hidden, n_classes,_char_size,_embbed_size,model_index,is_train):
    # Embedding Layer
    # _parm: char_size = 251(default)
    # _parm: config.embedding = 8(default)
    # char_embedding is tf.Variable size[251,8]
    with tf.name_scope("embedding-" + str(model_index)):
        embedding_W = tf.Variable(
            tf.random_uniform([_char_size, _embbed_size], -1.0, 1.0),
            name="Embedding_W" + "-%s" % model_index
        )

        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        embedded = tf.nn.embedding_lookup(embedding_W, input)
        #embedded_expand = tf.expand_dims(embedded, -1)
        #embedded_expand = tf.layers.batch_normalization(embedded_expand, training=is_train)
        #print(embedded_expand)

    with tf.name_scope("text-rnn-layer-" + str(model_index)):
        if model_index == 0:
            cell0 = tf.contrib.rnn.LSTMCell(n_hidden)
            outputs, states = tf.nn.dynamic_rnn(cell0, embedded, dtype=tf.float32)

        else:
            cell1 = tf.contrib.rnn.LSTMCell(n_hidden)
            outputs, states = tf.nn.dynamic_rnn(cell1, embedded, dtype=tf.float32)

        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]

    with tf.name_scope("Output-layer-" + str(model_index)):
        w = tf.Variable(tf.random_normal([n_hidden, n_classes]),dtype=tf.float32)
        b = tf.Variable(tf.random_normal([n_classes]),dtype=tf.float32)
        model = tf.nn.relu(tf.matmul(outputs, w) + b)

    return model


''' DROP OUT VER'''
# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os

import numpy as np
import tensorflow as tf
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from model.kin_dataset import KinQueryDataset


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output, feed_dict={x: preprocessed_data,keep_prob:1,batch_nom:False})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--dropout',type=float,default=0.5)
    args.add_argument('--batchnom',type=bool,default=True)
    args.add_argument('--lr', type=float, default=0.001)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = './sample_data/kin/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen # 8 x 400
    output_size = 1
    hidden_layer1_size = 512
    hidden_layer2_size = 256
    hidden_layer3_size = 128
    learning_rate = config.lr
    lr_decay = 0.99
    character_size = 251

    # Place holder
    x = tf.placeholder(tf.int32, [None, config.strmaxlen]) # 400
    y_ = tf.placeholder(tf.float32, [None, output_size]) # 1
    keep_prob = tf.placeholder(tf.float32)
    batch_nom = tf.placeholder(tf.bool)
    lr = tf.placeholder(tf.float32)

    # 임베딩
    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding])
    embedded = tf.nn.embedding_lookup(char_embedding, x)
    embedded = tf.layers.batch_normalization(embedded,training=batch_nom)
    # 첫 번째 레이어
    W1 = weight_variable([input_size, hidden_layer1_size])
    b1 = bias_variable([hidden_layer1_size])
    Layer1 = tf.matmul(tf.reshape(embedded, (-1, input_size)),
                       W1) + b1
    #Layer1 = tf.nn.dropout(Layer1,keep_prob=keep_prob)
    Layer1 = tf.layers.batch_normalization(Layer1,training=batch_nom)

    # 두 번째 (아웃풋) 레이어
    W2 = weight_variable([hidden_layer1_size, hidden_layer2_size])
    b2 = bias_variable([hidden_layer2_size])
    Layer2 = tf.matmul(Layer1, W2) + b2
    Layer2 = tf.sigmoid(Layer2)
    #Layer2 = tf.nn.dropout(Layer2, keep_prob=keep_prob)
    Layer2 = tf.layers.batch_normalization(Layer2, training=batch_nom)

    # layer-3
    W3 = weight_variable([hidden_layer2_size, hidden_layer3_size])
    b3 = bias_variable([hidden_layer3_size])
    Layer3 = tf.matmul(Layer2, W3) + b3
    Layer3 = tf.sigmoid(Layer3)
    #Layer3 = tf.nn.dropout(Layer3, keep_prob=keep_prob)
    Layer3 = tf.layers.batch_normalization(Layer3, training=batch_nom)

    # layer-4
    W4 = weight_variable([hidden_layer3_size, output_size])
    b4 = bias_variable([output_size])
    output = tf.matmul(Layer3, W4) + b4
    output = tf.sigmoid(output)


    # loss와 optimizer
    binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(output, 1e-10, 1.0))) - (1 - y_) * tf.log(
        tf.clip_by_value(1 - output, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(lr).minimize(binary_cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            avg_val_loss = 0.0
            s = np.random.permutation(dataset.queries.shape[0])
            dataset.queries = dataset.queries[s]
            dataset.labels = dataset.labels[s]
            dataset.queries = ((dataset.queries - np.mean(dataset.queries)) / np.std(dataset.queries))
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):

                test_idx = (int)(len(data) * 0.8)
                train_data = data[:test_idx]
                train_labels = labels[:test_idx]
                test_data = data[test_idx:]
                test_labels = labels[test_idx:]

                _, loss = sess.run([train_step, binary_cross_entropy],
                                   feed_dict={x: train_data, y_: train_labels,
                                              keep_prob:config.dropout,
                                              lr:learning_rate,
                                              batch_nom:config.batchnom
                                              })
                # Test Validation Set
                pred = sess.run(output, feed_dict={x: test_data,keep_prob:1,
                                                    batch_nom:False
                                                   })

                pred_clipped = np.array(pred > config.threshold, dtype=np.int)
                is_correct = tf.equal(pred_clipped, test_labels)
                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
                # Get Validation Loss
                val_loss = accuracy.eval(feed_dict={x: test_data,
                                                    y_: test_labels,keep_prob:1,
                                                    batch_nom:False
                                                    })
                print('Batch : ', i + 1, '/', one_batch_size, ', Batch Size:', one_batch_size,
                      ', BCE in this minibatch: ', float(loss), " Valid loss:",
                      float(val_loss))
                avg_loss += float((loss))
                avg_val_loss += float((val_loss))

            print('epoch:', epoch, ' train_loss:', float(avg_loss / (one_batch_size)), ' valid_loss:',
                  float(avg_val_loss / (one_batch_size)))
            learning_rate *= lr_decay
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/one_batch_size), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)



"""
Copyright 2018 NAVER Corp.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from model.kin_dataset import KinQueryDataset
from model.kin_cnn_models import text_cnn_ver_2


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output, feed_dict={x: preprocessed_data, is_training: False, keep_prob: 1})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다
    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def textCNNModel(_filter_sizes, _num_filters, _embedding_size, drop_out, model_index):
    # 모델의 specification
    # ====================================== MODEL ===========================================#
    #                           Text Classification using CNN

    l2_loss = tf.constant(0.0)

    # Embedding Layer
    # _parm: char_size = 251(default)
    # _parm: config.embedding = 8(default)
    # char_embedding is tf.Variable size[251,8]
    with tf.name_scope("embedding" + "-%s" % model_index):
        embedding_W = tf.Variable(
            tf.random_uniform([character_size, _embedding_size], -1.0, 1.0),
            name="Embedding_W" + "-%s" % model_index
        )

        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        embedded = tf.nn.embedding_lookup(embedding_W, x)
        embedded_expand = tf.expand_dims(embedded, -1)
        # embedded_expand = tf.nn.dropout(embedded_expand,drop_out)
        # batch normalization
        # embedded_expand = tf.layers.batch_normalization(embedded_expand,training=is_training)
        print(embedded_expand)

    # CNN-Clf Layer1
    # create convolution + maxpool layer
    num_of_filters = _num_filters  # 256
    filter_sizes = _filter_sizes  # [2,3,4,5]
    pooled_outputs = []

    # Convolution Layer
    for filter_size in filter_sizes:
        filter_name = "conv-maxpool-1-%s" % filter_size
        with tf.name_scope(filter_name + "-%s" % model_index):
            filter_shape = [filter_size, _embedding_size, 1, num_of_filters]
            Conv_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                 name="Conv_W-%s" % model_index)  # Conv's filter?
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
            h = tf.nn.dropout(h, drop_out)
            # h = tf.layers.batch_normalization(h,training=is_training)

            # Max pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, config.strmaxlen - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="MaxPool" + "-%s" % model_index
            )
            pooled = tf.nn.dropout(pooled, drop_out)
            # pooled = tf.layers.batch_normalization(pooled,training=is_training)
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_total_filters = num_of_filters * len(filter_sizes)  # 1 -> length of filter size
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_expand = tf.reshape(h_pool, [-1, num_total_filters])

    # Add Drop out

    with tf.name_scope("dropout" + "-%s" % model_index):
        h_drop = tf.nn.dropout(h_pool_expand, drop_out)
        # h_drop = tf.layers.batch_normalization(h_pool_expand,training=is_training)

    # Output layer
    with tf.name_scope("output-layer" + "-%s" % model_index):
        W2 = tf.get_variable(
            "W-out" + "-%s" % model_index,
            shape=[num_total_filters, L3_OUTPUT],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        B2 = tf.Variable(tf.constant(0.1, shape=[L3_OUTPUT]), name="B-out" + "-%s" % model_index)
        l2_loss += tf.nn.l2_loss(W2)
        l2_loss += tf.nn.l2_loss(B2)

        output = tf.nn.relu(tf.matmul(h_drop, W2) + B2)
        output = tf.nn.dropout(output, drop_out)
        # output = tf.layers.batch_normalization(output,training=is_training)
        print(output)

    return h_pool_expand


def textRNNModel(input, n_hidden, n_class):
    with tf.name_scope("rnn-layer"):
        input = tf.reshape([-1, tf.shape(input)])
        cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
        outputs, states = tf.nn.dynamic_rnn(cell, input, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]

    outputs = tf.reshape([-1, tf.shape(input)])

    with tf.name_scope("Output-layer"):
        w = tf.Variable(tf.random_normal([n_hidden, n_class]))
        b = tf.Variable(tf.random_normal([n_class]))
        model = tf.nn.sigmoid(tf.matmul(outputs, w) + b)

    return model


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--lr', type=float, default=0.001)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = './sample_data/kin/'

    L1_INPUT = config.embedding * config.strmaxlen  # 8 x 400
    H1_size = 1024
    H2_size = 256
    L3_OUTPUT = 50
    FIN_OUTPUT = 1
    learning_rate = config.lr
    learning_rate_tf = tf.placeholder(tf.float32, [], name="lr")
    train_decay = 0.99
    character_size = 251
    drop_out_val = 0.5
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    is_train = True
    # Input & Output layer
    # 'x' is sentence input layer(size 400). sentence data is a 400 max_len vector
    # and char2vec model return 'int32' vector
    x = tf.placeholder(tf.int32, [None, config.strmaxlen], name="input_x")  # 400

    # 'y' is output layer.
    # we will classify as binary (0 or 1)
    # so output size is one(1)
    y_ = tf.placeholder(tf.float32, [None, FIN_OUTPUT], name="output_y")  # 1

    """
    # combine all output layers
    outputs = []

    # RNN output node weights and biases
    num_hidden = 32
    num_classes = 1

    # Embedding Layer
    # _parm: char_size = 251(default)
    # _parm: config.embedding = 8(default)
    # char_embedding is tf.Variable size[251,8]
    with tf.name_scope("embedding"):
        embedding_W = tf.Variable(
            tf.random_uniform([character_size, config.embedding], -1.0, 1.0),
            name="Embedding_W"
        )

        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        intput = tf.nn.embedding_lookup(embedding_W, x)
        # embedded_expand = tf.expand_dims(embedded, -1)
        # embedded_expand = tf.nn.dropout(embedded_expand,drop_out)
        # batch normalization
        # embedded_expand = tf.layers.batch_normalization(embedded_expand,training=is_training)
        print(intput)
    """

    output = text_cnn_ver_2(_x=x, _character_size=character_size, _embedding_size=config.embedding,
                            drop_out=keep_prob, is_train=is_training, model_index=0)

    # output = textRNNModel(input=intput, n_hidden=num_hidden,n_class=num_classes)

    # outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[2, 3, 4, 5, 6, 3, 2], _embedding_size=16, drop_out=keep_prob,
    #                                model_index=0))
    # outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[2, 3, 3, 4, 4, 3, 2], _embedding_size=16, drop_out=keep_prob,
    #                                model_index=1))
    # outputs.append(textCNNModel(_num_filters=48, _filter_sizes=[4, 4, 4, 4], _embedding_size=16, drop_out=keep_prob,
    #                                model_index=2))
    # outputs.append(textCNNModel(_num_filters=48, _filter_sizes=[5, 5, 5, 5], _embedding_size=16, drop_out=keep_prob,
    #                                model_index=3))

    # outputs.append(textCNNModel(_num_filters=256, _filter_sizes=[2, 3, 4, 5], _embedding_size=16, drop_out=0.5,model_index=4))
    # outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[3, 4, 5, 6], _embedding_size=16, drop_out=0.7,model_index=5))
    # outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[4, 5, 6, 7], _embedding_size=16, drop_out=0.7,model_index=6))

    # output_concat = tf.concat(outputs,1)
    # total_output_len = len(outputs) * L3_OUTPUT
    # output_expand = tf.reshape(output_concat,[-1,total_output_len])
    # output_expand = tf.nn.dropout(output_concat, keep_prob)
    # output_pooled = tf.layers.batch_normalization(output_expand,training=is_training)
    # print(output_expand)

    # Rnn Final Layer
    # output = textRNNModel(output_expand,n_hidden=num_hidden,n_class=num_classes)

    """
    # Output layer
    with tf.name_scope("final-output-layer"):
        W_Fin = tf.get_variable(
            "W-final-out",
            shape=[total_output_len, FIN_OUTPUT],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        B_Fin = tf.Variable(tf.constant(0.1, shape=[FIN_OUTPUT]), name="B-final-out")
        output = tf.nn.sigmoid(tf.matmul(output_expand, W_Fin) + B_Fin)
        #output = tf.layers.batch_normalization(output,training=is_training)
        print(output)
    """

    # loss와 optimizer
    with tf.name_scope("loss-optimizer"):
        # Regularization..?
        # beta = 0.01
        binary_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=output, labels=y_
        ))
        # regularization = tf.nn.l2_loss(output)
        # binary_cross_entropy = tf.reduce_mean(binary_cross_entropy + beta*regularization)
        # binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(output,1e-10,1.0))) - (1-y_) * tf.log(tf.clip_by_value(1-output,1e-10,1.0)))
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_tf).minimize(binary_cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # ========================================================================================#

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len // config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            avg_val_loss = 0.0
            # Data nomalization
            s = np.random.permutation(dataset.queries.shape[0])
            dataset.queries = dataset.queries[s]
            dataset.labels = dataset.labels[s]
            dataset.queries = ((dataset.queries - np.mean(dataset.queries)) / np.std(dataset.queries))
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                # Divide Cross Validation Set
                test_idx = (int)(len(data) * 0.8)
                train_data = data[:test_idx]
                train_labels = labels[:test_idx]
                test_data = data[test_idx:]
                test_labels = labels[test_idx:]
                _, loss = sess.run([train_step, binary_cross_entropy],
                                   feed_dict={x: train_data, y_: train_labels, learning_rate_tf: learning_rate,
                                              is_training: True, keep_prob: drop_out_val})

                # Test Validation Set
                pred = sess.run(output, feed_dict={x: test_data, is_training: False, keep_prob: 1})
                pred_clipped = np.array(pred > config.threshold, dtype=np.int)
                is_correct = tf.equal(pred_clipped, test_labels)
                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
                # Get Validation Loss
                val_loss = accuracy.eval(feed_dict={x: test_data, y_: test_labels, is_training: False, keep_prob: 1})
                print('Batch : ', i + 1, '/', one_batch_size, ', Batch Size:', one_batch_size,
                      ', BCE in this minibatch: ', float(loss), " Valid loss:",
                      float(val_loss))
                avg_loss += float((loss))
                avg_val_loss += float((val_loss))

            print('epoch:', epoch, ' train_loss:', float(avg_loss / (one_batch_size)), ' valid_loss:',
                  float(avg_val_loss / (one_batch_size)))
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss / one_batch_size), step=epoch)
            learning_rate = learning_rate * train_decay
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    else:
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)



"""
Copyright 2018 NAVER Corp.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os

import numpy as np
import tensorflow as tf
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from model.kin_dataset import KinQueryDataset


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output, feed_dict={x: preprocessed_data, keep_prob: 1})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다
    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def textCNNModel(_filter_sizes, _num_filters, _embedding_size,drop_out,model_index):
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
            tf.random_uniform([character_size, _embedding_size], -1.0, 1.0),
            name="Embedding_W"+ "-%s" % model_index
        )

        # embedded is a embedding neural net which has input as 'char_embedding' & input sentence 'x'
        embedded = tf.nn.embedding_lookup(embedding_W, x)
        embedded_expand = tf.expand_dims(embedded, -1)
        embedded_expand = tf.nn.dropout(embedded_expand,drop_out)
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
            filter_shape = [filter_size, _embedding_size, 1, num_of_filters]
            Conv_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Conv_W-%s" % model_index)  # Conv's filter?
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
            h = tf.nn.dropout(h,drop_out)

            # Max pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, config.strmaxlen - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="MaxPool" + "-%s" % model_index
            )
            pooled = tf.nn.dropout(pooled,drop_out)
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
        W2 = tf.get_variable(
            "W-out"+ "-%s" % model_index,
            shape=[num_total_filters, L3_OUTPUT],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        B2 = tf.Variable(tf.constant(0.1, shape=[L3_OUTPUT]), name="B-out"+ "-%s" % model_index)
        l2_loss += tf.nn.l2_loss(W2)
        l2_loss += tf.nn.l2_loss(B2)

        output = tf.nn.relu(tf.matmul(h_drop, W2) + B2)
        output = tf.nn.dropout(output,drop_out)
        print(output)

    return output

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=16)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--lr',type=float,default=0.002)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = './sample_data/kin/'

    L1_INPUT = config.embedding * config.strmaxlen  # 8 x 400
    H1_size = 1024
    H2_size = 256
    L3_OUTPUT = 5
    FIN_OUTPUT = 1
    learning_rate = config.lr
    learning_rate_tf = tf.placeholder(tf.float32,[],name="lr")
    train_decay = 0.99
    character_size = 251
    drop_out_val = 0.7
    keep_prob = tf.placeholder(tf.float32)

    # Input & Output layer
    # 'x' is sentence input layer(size 400). sentence data is a 400 max_len vector
    # and char2vec model return 'int32' vector
    x = tf.placeholder(tf.int32, [None, config.strmaxlen], name="input_x")  # 400

    # 'y' is output layer.
    # we will classify as binary (0 or 1)
    # so output size is one(1)
    y_ = tf.placeholder(tf.float32, [None, FIN_OUTPUT], name="output_y")  # 1

    # combine all output layers
    outputs = []

    outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[2, 3, 4, 5, 4, 3, 2], _embedding_size=8, drop_out=keep_prob,
                                    model_index=0))
    outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[4, 6, 6, 4], _embedding_size=16, drop_out=keep_prob,
                                    model_index=1))
    outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[5, 6, 7, 8], _embedding_size=16, drop_out=keep_prob,
                                    model_index=2))
    #outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[5, 5, 5, 5], _embedding_size=16, drop_out=0.7,
    #                                model_index=3))

    #outputs.append(textCNNModel(_num_filters=256, _filter_sizes=[2, 3, 4, 5], _embedding_size=16, drop_out=0.5,model_index=4))
    #outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[3, 4, 5, 6], _embedding_size=16, drop_out=0.7,model_index=5))
    #outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[4, 5, 6, 7], _embedding_size=16, drop_out=0.7,model_index=6))

    output_concat = tf.concat(outputs,1)
    total_output_len = len(outputs) * L3_OUTPUT
    output_expand = tf.reshape(output_concat,[-1,total_output_len])
    output_expand = tf.nn.dropout(output_expand, keep_prob)
    print(output_expand)

    # Output layer
    with tf.name_scope("final-output-layer"):
        W_Fin = tf.get_variable(
            "W-final-out",
            shape=[total_output_len, FIN_OUTPUT],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        B_Fin = tf.Variable(tf.constant(0.1, shape=[FIN_OUTPUT]), name="B-final-out")
        output = tf.nn.sigmoid(tf.matmul(output_expand, W_Fin) + B_Fin)
        print(output)


    # loss와 optimizer
    with tf.name_scope("loss-optimizer"):
        # Regularization..?
        #beta = 0.01
        #binary_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #    logits=output, labels=y_
        #))
        #regularization = tf.nn.l2_loss(output)
        #binary_cross_entropy = tf.reduce_mean(binary_cross_entropy + beta*regularization)
        binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(output,1e-10,1.0))) - (1-y_) * tf.log(tf.clip_by_value(1-output,1e-10,1.0)))
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate_tf).minimize(binary_cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # ========================================================================================#

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)


    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            avg_val_loss = 0.0
            # Data nomalization
            s = np.random.permutation(dataset.queries.shape[0])
            dataset.queries = dataset.queries[s]
            dataset.labels = dataset.labels[s]
            dataset.queries = ((dataset.queries - np.mean(dataset.queries)) / np.std(dataset.queries))
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                # Divide Cross Validation Set
                test_idx = (int)(len(data) * 0.8)
                train_data = data[:test_idx]
                train_labels = labels[:test_idx]
                test_data = data[test_idx:]
                test_labels = labels[test_idx:]
                _, loss = sess.run([train_step, binary_cross_entropy], feed_dict={x: train_data, y_: train_labels, learning_rate_tf: learning_rate,keep_prob: drop_out_val})

                # Test Validation Set
                pred = sess.run(output, feed_dict={x: test_data, keep_prob: 1})
                pred_clipped = np.array(pred > config.threshold, dtype=np.int)
                is_correct = tf.equal(pred_clipped, test_labels)
                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
                # Get Validation Loss
                val_loss = accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1})
                print('Batch : ', i + 1, '/', one_batch_size, ', Batch Size:', one_batch_size , ', BCE in this minibatch: ', float(loss), " Valid loss:",
                      float(val_loss))
                avg_loss += float((loss))
                avg_val_loss += float((val_loss))



            print('epoch:', epoch, ' train_loss:', float(avg_loss / (one_batch_size)), ' valid_loss:',
                  float(avg_val_loss / (one_batch_size)))
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/one_batch_size), step=epoch)
            learning_rate = learning_rate * train_decay
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    else:
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)


# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from model.kin_dataset import KinQueryDataset, preprocess

from model.kin_cnn_models import text_cnn_ver_1


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output, feed_dict={x: preprocessed_data})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--lr', type=float, default=0.001)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = './sample_data/kin/'

    L1_INPUT = config.embedding * config.strmaxlen  # 8 x 400
    H1_size = 1024
    H2_size = 256
    L3_OUTPUT = 5
    FIN_OUTPUT = 1
    beta = 0.01
    # learning_rate = config.lr
    character_size = 251
    learning_rate = 0.001
    train_decay = 0.99

    # Input & Output layer
    # 'x' is sentence input layer(size 400). sentence data is a 400 max_len vector
    # and char2vec model return 'int32' vector
    x = tf.placeholder(tf.int32, [None, config.strmaxlen], name="input_x")  # 400

    # 'y' is output layer.
    # we will classify as binary (0 or 1)
    # so output size is one(1)
    y_ = tf.placeholder(tf.float32, [None, FIN_OUTPUT], name="output_y")  # 1

    # outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[4, 4, 4, 4], _embedding_size=8, drop_out=0.7,model_index=2))
    # outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[5, 5, 5, 5], _embedding_size=8, drop_out=0.7,model_index=3))
    # outputs.append(textCNNModel(_num_filters=256, _filter_sizes=[2, 3, 4, 5], _embedding_size=16, drop_out=0.5,model_index=4))
    # outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[3, 4, 5, 6], _embedding_size=16, drop_out=0.7,model_index=5))
    # outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[4, 5, 6, 7], _embedding_size=16, drop_out=0.7,model_index=6))

    # output_concat = tf.concat(outputs,1)
    # total_output_len = len(outputs) * L3_OUTPUT
    # output_expand = tf.reshape(output_concat,[-1,total_output_len])
    # print(output_expand)

    # Output layer
    # with tf.name_scope("final-output-layer"):
    #    W_Fin = tf.get_variable(
    #        "W-final-out",
    #        shape=[total_output_len, FIN_OUTPUT],
    #        initializer=tf.contrib.layers.xavier_initializer()
    #    )

    #    B_Fin = tf.Variable(tf.constant(0.1, shape=[FIN_OUTPUT]), name="B-final-out")

    #    output = tf.nn.sigmoid(tf.matmul(output_expand, W_Fin) + B_Fin)

    outputs = []
    output_size = 5
    outputs.append(
        text_cnn_ver_1(_x=x, _filter_sizes=[2, 3, 4, 5, 6, 3, 2], _character_size=character_size, _num_filters=64
                       , _embedding_size=config.embedding, drop_out=0.7, model_index=0,
                       strmaxlen=config.strmaxlen, n_classes=output_size))
    '''
    outputs.append(text_cnn_ver_1(_x=x, _filter_sizes=[2, 3, 3, 4, 4, 3, 2], _character_size=character_size, _num_filters=64
                                  , _embedding_size=config.embedding, drop_out=0.7, model_index=1,
                                  strmaxlen=config.strmaxlen,final_output_num=output_size))
    '''
    output_concat = tf.concat(outputs, 1)
    total_output_len = len(outputs) * output_size
    output_expand = tf.reshape(output_concat, [-1, total_output_len])
    print(output_expand)

    # Output layer
    final_output_size = 1
    with tf.name_scope("final-output-layer"):
        W_Fin = tf.get_variable(
            "W-final-out",
            shape=[total_output_len, final_output_size],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        B_Fin = tf.Variable(tf.constant(0.1, shape=[final_output_size]), name="B-final-out")

        output = tf.nn.sigmoid(tf.matmul(output_expand, W_Fin) + B_Fin)

    # loss와 optimizer
    with tf.name_scope("loss-optimizer"):
        binary_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=output, labels=y_
        ))
        # Regularization..?
        # regularization = tf.nn.l2_loss(output)
        # binary_cross_entropy = tf.reduce_mean(binary_cross_entropy + beta*regularization)
        # binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(output,1e-10,1.0))) - (1-y_) * tf.log(tf.clip_by_value(1-output,1e-10,1.0)))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(binary_cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # ========================================================================================#

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len // config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            avg_val_loss = 0.0
            # s = np.random.permutation(dataset.queries.shape[0])
            # dataset.queries = dataset.queries[s]
            # dataset.labels = dataset.labels[s]
            # dataset.queries = ((dataset.queries - np.mean(dataset.queries)) / np.std(dataset.queries))
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                # Divide Cross Validation Set
                test_idx = (int)(len(data) * 0.8)
                train_data = data[:test_idx]
                train_labels = labels[:test_idx]
                test_data = data[test_idx:]
                test_labels = labels[test_idx:]
                _, loss = sess.run([train_step, binary_cross_entropy], feed_dict={x: train_data, y_: train_labels})

                # Test Validation Set
                pred = sess.run(output, feed_dict={x: test_data})
                pred_clipped = np.array(pred > config.threshold, dtype=np.int)
                is_correct = tf.equal(pred_clipped, test_labels)
                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

                # Get Validation Loss
                val_loss = accuracy.eval(feed_dict={x: test_data, y_: test_labels})
                print('Batch : ', i + 1, '/', one_batch_size, ', BCE in this minibatch: ', float(loss), " Valid loss:",
                      float(val_loss))
                avg_loss += float((loss))
                avg_val_loss += float((val_loss))

            print('epoch:', epoch, ' train_loss:', float(avg_loss / (one_batch_size)), ' valid_loss:',
                  float(avg_val_loss / (one_batch_size)))
            # print('Learning rate: ' , learning_rate)
            # learning_rate = learning_rate * train_decay
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss / (one_batch_size)), step=epoch)

            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)



    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    else:
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)




