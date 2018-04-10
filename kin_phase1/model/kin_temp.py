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




