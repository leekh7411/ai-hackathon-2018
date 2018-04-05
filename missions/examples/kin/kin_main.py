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
from kin_dataset import KinQueryDataset, preprocess
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

            # Max pooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, config.strmaxlen - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="MaxPool" + "-%s" % model_index
            )
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

        output = tf.nn.sigmoid(tf.matmul(h_drop, W2) + B2)
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
    args.add_argument('--epochs', type=int, default=500)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=16)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--lr',type=float,default=0.001)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = './sample_data/kin/'

    L1_INPUT = config.embedding * config.strmaxlen  # 8 x 400
    H1_size = 1024
    H2_size = 256
    L3_OUTPUT = 5
    FIN_OUTPUT = 1
    learning_rate = config.lr
    character_size = 251

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
    outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[2, 3, 4, 5, 6, 3, 2], _embedding_size=18, drop_out=0.7,model_index=0))
    outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[2, 3, 3, 4, 4, 3, 2], _embedding_size=18, drop_out=0.7,model_index=1))
    #outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[4, 4, 4, 4], _embedding_size=8, drop_out=0.7,model_index=2))
    #outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[5, 5, 5, 5], _embedding_size=8, drop_out=0.7,model_index=3))
    #outputs.append(textCNNModel(_num_filters=256, _filter_sizes=[2, 3, 4, 5], _embedding_size=16, drop_out=0.5,model_index=4))
    #outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[3, 4, 5, 6], _embedding_size=16, drop_out=0.7,model_index=5))
    #outputs.append(textCNNModel(_num_filters=64, _filter_sizes=[4, 5, 6, 7], _embedding_size=16, drop_out=0.7,model_index=6))

    output_concat = tf.concat(outputs,1)
    total_output_len = len(outputs) * L3_OUTPUT
    output_expand = tf.reshape(output_concat,[-1,total_output_len])
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
        binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(output,1e-10,1.0))) - (1-y_) * tf.log(tf.clip_by_value(1-output,1e-10,1.0)))
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
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(_batch_loader(dataset, config.batch)):
                _, loss = sess.run([train_step, binary_cross_entropy],
                                   feed_dict={x: data, y_: labels})
                print('Batch : ', i + 1, '/', one_batch_size,
                      ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)
            print('epoch:', epoch, ' train_loss:', float(avg_loss/one_batch_size))
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/one_batch_size), step=epoch)
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