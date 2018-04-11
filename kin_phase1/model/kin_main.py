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
from kin_cnn_models import text_clf_model_ver3, text_clf_model_ver2

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
        preprocessed_data1,preprocessed_data2 = preprocess(raw_data, config.strmaxlen)

        # data normalization
        preprocessed_data1 = data_normalization(preprocessed_data1)
        preprocessed_data2 = data_normalization(preprocessed_data2)

        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output, feed_dict={
                                input1: preprocessed_data1,
                                input2: preprocessed_data2,
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
    args.add_argument('--lr',type=float,default=0.001)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '/home/leekh7411/PycharmProject/ai-hackathon-2018/kin_phase1/sample_data/kin/'

    L1_INPUT = config.embedding * config.strmaxlen  # 8 x 400
    FIN_OUTPUT = 1
    learning_rate = config.lr
    learning_rate_tf = tf.placeholder(tf.float32,[],name="lr")
    train_decay = 0.99
    character_size = 251
    drop_out_val = 0.5
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    is_train = True
    is_data_norm = True
    n_classes = 128
    rnn_h_num = 128

    # Input & Output layer
    # 'x' is sentence input layer(size 400). sentence data is a 400 max_len vector
    # and char2vec model return 'int32' vector
    input1 = tf.placeholder(tf.int32, [None, config.strmaxlen], name="input1_x")  # 400
    input2 = tf.placeholder(tf.int32, [None, config.strmaxlen], name="input2_x")

    # 'y' is output layer.
    # we will classify as binary (0 or 1)
    # so output size is one(1)
    y_ = tf.placeholder(tf.float32, [None, FIN_OUTPUT], name="output_y")  # 1


    output = text_clf_model_ver2(
        input1 = input1,
        input2 = input2,
        char_size = character_size,
        embedding_size= config.embedding,
        is_train=is_train,
        keep_prob= keep_prob
    )

    # loss와 optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        with tf.name_scope("loss-optimizer"):
            binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(output,1e-10,1.0))) - (1-y_) * tf.log(tf.clip_by_value(1-output,1e-10,1.0)))
            train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(binary_cross_entropy)


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

            dataset.queries1 = data_normalization(dataset.queries1)
            dataset.queries2 = data_normalization(dataset.queries2)
            for i, (data1,data2, labels) in enumerate(_batch_loader(dataset, config.batch)):
                # Divide Cross Validation Set
                test_idx = (int)(len(labels) * 0.8)
                train1_data = data1[:test_idx]
                train2_data = data2[:test_idx]

                test_data1 = data1[test_idx:]
                test_data2 = data2[test_idx:]

                train_labels = labels[:test_idx]
                test_labels = labels[test_idx:]

                #shuffle
                s = np.random.permutation(train_labels.shape[0])
                train1_data = train1_data[s]
                train2_data = train2_data[s]
                train_labels = train_labels[s]

                _, loss = sess.run([train_step, binary_cross_entropy],
                                   feed_dict={
                                       input1: train1_data,
                                       input2: train2_data,
                                       y_: train_labels,
                                       learning_rate_tf: learning_rate,
                                       is_training: True,
                                       keep_prob: drop_out_val
                                   })

                # Test Validation Set
                pred = sess.run(output, feed_dict={input1: test_data1,input2: test_data2,y_:test_labels ,is_training: False, keep_prob:1})
                pred_clipped = np.array(pred > config.threshold, dtype=np.int)
                is_correct = tf.equal(pred_clipped, test_labels)
                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

                # Get Validation Loss
                val_acc = accuracy.eval(
                    feed_dict={input1: test_data1,
                               input2: test_data2,
                               y_: test_labels,
                               is_training: False,
                               keep_prob:1})

                val_loss = sess.run(binary_cross_entropy,
                                    feed_dict={input1: test_data1,
                                               input2: test_data2,
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



