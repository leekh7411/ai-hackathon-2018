import argparse
import os
from kin_dataset import KinQueryDataset, preprocess
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, Conv1D, Conv2D, Embedding, GlobalMaxPooling1D
from keras.layers import MaxPooling1D, MaxPooling2D, Flatten, LSTM
from sklearn.model_selection import train_test_split

'''
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
from dataset import KinQueryDataset, preprocess
'''


def numeric_input_sigmoid_clf_model(input_dim=400):
    model = Sequential()
    model.add(Dense(756, input_dim=input_dim, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    return model

def text_input_binary_clf_RNN_CNN_model(input_length=400):
    model = Sequential()
    model.add(Embedding(20000, 128, input_length=input_length))
    model.add(Dropout(0.2))
    model.add(Conv1D(256,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    return model


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


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        '''
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))
        '''
        # save model in dir_name
        model.save_weights(dir_name + 'model.h5')
        print("Model saved")

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        """
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        """
        model.load_weights(dir_name + 'model.h5')
        print('Model loaded')

    # Predict and Test
    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)

        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        preprocessed_data = cover_numpy_arr(preprocessed_data)
        pred = sess.predict(preprocessed_data)

        # pred = sess.run(output_sigmoid, feed_dict={x: preprocessed_data})
        clipped = np.array(pred > config.threshold, dtype=np.int)

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다

        return list(zip(pred.flatten(), clipped.flatten()))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    # nsml.bind(save=save, load=load, infer=infer)


def cover_numpy_arr(data):
    new_data = np.array(data)
    new_data = np.reshape(new_data, [1, len(new_data)])
    return new_data


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--embedding', type=int, default=8)
    args.add_argument('--threshold', type=float, default=0.5)
    config = args.parse_args()

    # REMOVE  #
    # if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
    DATASET_PATH = './sample_data/kin/'

    # 모델의 specification
    input_size = config.embedding * config.strmaxlen
    output_size = 1
    hidden_layer_size = 200
    learning_rate = 0.001
    character_size = 251

    # Get Text CLF Model
    #model = text_input_binary_clf_RNN_CNN_model(input_length=config.strmaxlen)
    model = numeric_input_sigmoid_clf_model(input_dim=config.strmaxlen)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=model, config=config)

    # DONOTCHANGE: Reserved for nsml
    # if config.pause: REMOVE#
    # nsml.paused(scope=locals()) REMOVE#

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        one_batch_size = dataset_len // config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.

        hist = model.fit(dataset.queries, dataset.labels, epochs=config.epochs, batch_size=10)
        # DONOTCHANGE (You can decide how often you want to save the model)
        # nsml.save(1)
        model.save_weights(DATASET_PATH + 'model.h5')


    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        print("Local testing....")
        print("..................Loading Model.......")
        tf.keras.models.load_model(
            DATASET_PATH,
            custom_objects=None,
            compile=True
        )
        print(".......Finish!")
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            # temp_res = nsml.infer(batch) REMOVE#
            temp_res = 0
            res += temp_res
        print(res)
