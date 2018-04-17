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
from kin_dataset import KinQueryDataset, preprocess,preprocess2,data_augmentation,preprocess_origin
from kin_model import text_clf_ensemble_model
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
        preprocessed_data1, preprocessed_data2 = preprocess2(raw_data, config.strmaxlen,test_data=False)
        #preprocessed_data = preprocess_origin(raw_data, config.strmaxlen,test_data=False)

        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        infer_preds = []
        for output in outputs:
            infer_preds.append(sess.run(output, feed_dict={
                                    input1: preprocessed_data1,
                                    input2: preprocessed_data2,
                                    is_training: False,
                                    keep_prob:1}))

        infer_pred = tf.concat(infer_preds, axis=1)
        infer_pred = tf.reduce_mean(infer_pred, axis=1, keep_dims=True)
        infer_pred = sess.run(infer_pred)
        clipped = np.array((infer_pred) > config.threshold, dtype=np.int)

        # clipped = np.array(infer_pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        return list(zip(infer_pred.flatten(), clipped.flatten()))

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
    #    config_proto = tf.ConfigProto()
    #    config_proto.gpu_options.allow_growth = True

        args = argparse.ArgumentParser()
        # DONOTCHANGE: They are reserved for nsml
        args.add_argument('--mode', type=str, default='train')
        args.add_argument('--pause', type=int, default=0)
        args.add_argument('--iteration', type=str, default='0')

        # User options
        args.add_argument('--output', type=int, default=1)
        args.add_argument('--epochs', type=int, default=200)
        args.add_argument('--batch', type=int, default=2000)
        args.add_argument('--strmaxlen', type=int, default=400)
        args.add_argument('--w2v_size',type=int, default=16)
        args.add_argument('--embedding', type=int, default=8) # more bigger?
        args.add_argument('--threshold', type=float, default=0.5)
        args.add_argument('--lr',type=float,default=0.0005)
        config = args.parse_args()

        if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
            # This path have to be changed!
            DATASET_PATH = '/home/leekh7411/PycharmProject/ai-hackathon-2018/kin_phase1/sample_data/kin/'

        # Parameters for model configuration
        L1_INPUT = config.embedding * config.strmaxlen  # 8 x 400
        FIN_OUTPUT = 1
        learning_rate = config.lr
        learning_rate_tf = tf.placeholder(tf.float32,[],name="lr")
        train_decay = 0.99
        character_size = 251
        w2v_size = config.w2v_size
        drop_out_val = 0.8
        keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool)
        is_train = True
        is_data_norm = False
        n_classes = 32
        beta = 0.1

        # Input & Output layer
        input1 = tf.placeholder(tf.int32, [None, config.strmaxlen], name="input-x1")
        input2 = tf.placeholder(tf.int32, [None, config.strmaxlen], name="input-x2")
        y_ = tf.placeholder(tf.float32, [None, FIN_OUTPUT], name="output-y")


        # Add models for ensemble prediction
        outputs = []
        ensemble_size = 10
        for i in range(ensemble_size):
            outputs.append(text_clf_ensemble_model(input1, input2, character_size, config.embedding, is_train, keep_prob, n_classes,i))


        # Make each model's loss and optimizer(train_step)
        with tf.name_scope("loss-optimizer"):
            # Binary Cross Entropy
            def binary_cross_entropy_loss(y_,output):
                return tf.reduce_mean(-(y_ * tf.log(tf.clip_by_value(output, 1e-10, 1.0))) - (1 - y_) * tf.log(tf.clip_by_value(1 - output, 1e-10, 1.0)))

            bce_loss = []
            for out in outputs:
                bce_loss.append(binary_cross_entropy_loss(y_,out))

            train_steps = []
            for loss in bce_loss:
                train_steps.append(tf.train.AdamOptimizer(learning_rate=learning_rate_tf).minimize(loss))

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

                # Shuffle train data to prevent overfitting
                s = np.random.permutation(dataset.labels.shape[0])
                dataset.queries1 = dataset.queries1[s]
                dataset.queries2 = dataset.queries2[s]
                dataset.labels = dataset.labels[s]

                #test_data1 = dataset.queries1_test
                #test_data2 = dataset.queries2_test
                #test_labels = dataset.labels_test

                for i, (data1,data2,labels) in enumerate(_batch_loader(dataset, config.batch)):

                    # Divide Cross Validation Set
                    # *This validation is meaningless! because of all data will be shuffled
                    test_idx = (int)(len(labels) * 0.95)
                    train_data1 = data1[:test_idx]
                    train_data2 = data2[:test_idx]
                    test_data1 = data1[test_idx:]
                    test_data2 = data2[test_idx:]
                    train_labels = labels[:test_idx]
                    test_labels = labels[test_idx:]


                    # Test Validation Set
                    # For ensemble, test each models
                    def predict(output,test_data1,test_data2,is_train,_keep_prob):
                        pred = sess.run(output, feed_dict={input1: test_data1,input2: test_data2,
                                                        is_training: is_train, keep_prob: _keep_prob})
                        pred_clipped = np.array(pred > config.threshold, dtype=np.float32)
                        return pred_clipped

                    preds = []
                    for out in outputs:
                        preds.append(predict(out, test_data1, test_data2, False, 1))

                    # concat all predicted results([0.,1.,0.,1.,..],[1.,0.,1.,...],...) <- float data
                    pred = tf.concat(preds,axis=1)

                    # sum and mean all row data
                    pred = tf.reduce_mean(pred,axis=1,keep_dims=True)

                    # if five models result's is 0.8
                    # --> [1,1,1,1,0] --> sum(4) --> mean(4/5) --> 0.8 --> threshold(0.5) --> 1
                    # ensemble's result is '1'
                    pred = np.array(sess.run(pred) > config.threshold, dtype=np.int)
                    is_correct = tf.equal(pred, test_labels)
                    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

                    # Get Validation Loss
                    val_acc = accuracy.eval(
                        feed_dict={
                                   input1: test_data1,input2: test_data2,
                                   y_: test_labels,
                                   is_training: False,
                                   keep_prob: 1})



                    # Training Section
                    ensemble_loss = 0.
                    for train, bce in zip(train_steps,bce_loss):
                        _, loss = sess.run([train, bce],
                                           feed_dict={
                                               input1: data1,input2: data2,
                                               y_: labels,
                                               learning_rate_tf: learning_rate,
                                               is_training: True,
                                               keep_prob: drop_out_val
                                           })
                        ensemble_loss += loss
                    ensemble_loss /= len(bce_loss)

                    nsml.report(summary=True, scope=locals(), epoch=epoch * one_batch_size + i, epoch_total=config.epochs * one_batch_size,
                                train__loss=float(ensemble_loss), step=epoch * one_batch_size + i)

                    print('Batch : ', i + 1, '/', one_batch_size, ', Batch Size:', one_batch_size ,
                          'BCE in this minibatch: ', float(ensemble_loss),
                          "Valid score:", float(val_acc) * 100,
                          "Learning_rate:", (learning_rate))
                    avg_loss += float((ensemble_loss))
                    avg_val_acc += float((val_acc))
                print('========================================================================================')
                print('epoch:', epoch, '\ntrain_loss:', float(avg_loss / (one_batch_size)),'\nvalid_acc:',
                      float(avg_val_acc / (one_batch_size)) * 100)


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



