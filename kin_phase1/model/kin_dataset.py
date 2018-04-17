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


import os
import tensorflow as tf
import numpy as np
import time
import math
from kin_kor_char_parser import decompose_str_as_one_hot
LOCAL_DATASET_PATH = '../sample_data/kin/'
from soy.soy.nlp.tokenizer import CohesionTokenizer, RegexTokenizer
from gensim.models import Word2Vec

class KinQueryDataset:
    """
        지식인 데이터를 읽어서, tuple (데이터, 레이블)의 형태로 리턴하는 파이썬 오브젝트 입니다.
    """
    def __init__(self, dataset_path: str, max_length: int):
        """

        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이 400
        """
        # 데이터, 레이블 각각의 경로
        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')

        # 지식인 데이터를 읽고 preprocess까지 진행합니다
        self.test_idx = -1
        with open(queries_path, 'rt', encoding='utf8') as f:
            #self.queries1, self.queries2,self.queries1_test,self.queries2_test,self.test_idx = preprocess2(f.readlines(), max_length,test_data=True)
            self.queries1, self.queries2= preprocess2(f.readlines(), max_length,test_data=False)
            #self.queries,self.queries_test,self.test_idx = preprocess_origin(f.readlines(),max_length,test_data=True)

        # 지식인 레이블을 읽고 preprocess까지 진행합니다.
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])
            if self.test_idx != -1:
                self.labels_test = self.labels[self.test_idx:]
                self.labels = self.labels[:self.test_idx]
                print("test data splited size %d" % self.test_idx)


    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.queries1[idx], self.queries2[idx] ,self.labels[idx]

def add_noise(query):
    query = query + (query * (0.001) * ((np.random.rand(1) - 0.5)))
    query = np.rint(query)
    query = query.astype(np.int32)
    return query

def data_augmentation(queries1,queries2,labels):
    # Add noise in query data
    def get_noised_queries(queries):
        # Expand query numpy array size
        q_expand = np.zeros((len(queries) * 2, len(queries[0])), dtype=np.int32)
        np.random.seed(int(time.time()))
        for i in range(len(q_expand)):
            if i < len(queries):
                q_expand[i] = queries[i]
            else:
                noised_val = add_noise(queries[i - len(queries)])
                q_expand[i] = noised_val

        return  q_expand

    def get_double_labels(labels):
        l_expand = np.zeros((len(labels) * 2,1), dtype=np.int32)
        for i in range(len(l_expand)):
            if i < len(labels):
                l_expand[i] = labels[i]
            else:
                l_expand[i] = labels[i - len(labels)]

        return l_expand

    q1_expand = get_noised_queries(queries1)
    q2_expand = get_noised_queries(queries2)
    l_expand = get_double_labels(labels)
    return q1_expand, q2_expand, l_expand

def preprocess2(data: list, max_length: int, test_data: bool):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """
    query1 =[]
    query2 =[]
    for d in data:
        q1,q2 = d.split('\t')
        query1.append(q1)
        query2.append(q2.replace('\n',''))

    vectorized_data1 = [decompose_str_as_one_hot(datum, warning=False) for datum in query1]
    vectorized_data2 = [decompose_str_as_one_hot(datum, warning=False) for datum in query2]

    if test_data :
        data_size = (len(data))
        test_size = (int)(data_size * 0.03)
        train_size = data_size - test_size
        zero_padding1 = np.zeros((train_size, max_length), dtype=np.int32)
        zero_padding2 = np.zeros((train_size, max_length), dtype=np.int32)
        zero_padding1_test = np.zeros((test_size, max_length), dtype=np.int32)
        zero_padding2_test = np.zeros((test_size, max_length), dtype=np.int32)
        for idx, seq in enumerate(vectorized_data1):
            if idx < train_size:
                length = len(seq)
                if length >= max_length:
                    length = max_length
                    zero_padding1[idx, :length] = np.array(seq)[:length]
                else:
                    zero_padding1[idx, :length] = np.array(seq)
            else:
                length = len(seq)
                if length >= max_length:
                    length = max_length
                    zero_padding1_test[idx - train_size, :length] = np.array(seq)[:length]
                else:
                    zero_padding1_test[idx - train_size, :length] = np.array(seq)

        for idx, seq in enumerate(vectorized_data2):
            if idx < train_size:
                length = len(seq)
                if length >= max_length:
                    length = max_length
                    zero_padding2[idx, :length] = np.array(seq)[:length]
                else:
                    zero_padding2[idx, :length] = np.array(seq)
            else:
                length = len(seq)
                if length >= max_length:
                    length = max_length
                    zero_padding2_test[idx - train_size, :length] = np.array(seq)[:length]
                else:
                    zero_padding2_test[idx - train_size, :length] = np.array(seq)

        return zero_padding1,zero_padding2, zero_padding1_test,zero_padding2_test, train_size

    else:
        data_size = (len(data))
        test_size = (int)(data_size * 0.03)
        train_size = data_size - test_size
        zero_padding1 = np.zeros((data_size, max_length), dtype=np.int32)
        zero_padding2 = np.zeros((data_size, max_length), dtype=np.int32)
        for idx, seq in enumerate(vectorized_data1):
            length = len(seq)
            if length >= max_length:
                length = max_length
                zero_padding1[idx, :length] = np.array(seq)[:length]
            else:
                zero_padding1[idx, :length] = np.array(seq)

        for idx, seq in enumerate(vectorized_data2):
            length = len(seq)
            if length >= max_length:
                length = max_length
                zero_padding2[idx, :length] = np.array(seq)[:length]
            else:
                zero_padding2[idx, :length] = np.array(seq)


        return zero_padding1, zero_padding2


def preprocess_origin(data: list, max_length: int,test_data: bool):
    """
     입력을 받아서 딥러닝 모델이 학습 가능한 포맷으로 변경하는 함수입니다.
     기본 제공 알고리즘은 char2vec이며, 기본 모델이 MLP이기 때문에, 입력 값의 크기를 모두 고정한 벡터를 리턴합니다.
     문자열의 길이가 고정값보다 길면 긴 부분을 제거하고, 짧으면 0으로 채웁니다.

    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """




    vectorized_data = [decompose_str_as_one_hot(datum, warning=False) for datum in data]
    if test_data :
        data_size = (len(data))
        test_size = (int)(data_size * 0.03)
        train_size = data_size - test_size
        zero_padding = np.zeros((train_size, max_length), dtype=np.int32)
        zero_padding_test=  np.zeros((test_size,max_length),dtype=np.int32)
        for idx, seq in enumerate(vectorized_data):
            if idx < train_size:
                length = len(seq)
                if length >= max_length:
                    length = max_length
                    zero_padding[idx, :length] = np.array(seq)[:length]
                else:
                    zero_padding[idx, :length] = np.array(seq)
            else:
                length = len(seq)
                if length >= max_length:
                    length = max_length
                    zero_padding_test[idx - train_size, :length] = np.array(seq)[:length]
                else:
                    zero_padding_test[idx - train_size, :length] = np.array(seq)

        return zero_padding, zero_padding_test, train_size
    else:
        zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
        for idx, seq in enumerate(vectorized_data):
            length = len(seq)
            if length >= max_length:
                length = max_length
                zero_padding[idx, :length] = np.array(seq)[:length]
            else:
                zero_padding[idx, :length] = np.array(seq)


        return zero_padding



def preprocess(data: list, max_length: int):
    q_words = []

    vector_size = 16
    tokenizer = RegexTokenizer()
    for d in data:
        q_words.append(tokenizer.tokenize(d))

    model = Word2Vec(q_words,size=vector_size,window=2,min_count=2,iter=100,sg=1)
    vocab_len = len(model.wv.vocab)
    print("word2vec -> vector size :", vector_size)
    print("word2vec -> vocab  size :", vocab_len)

    zero_padding = np.zeros((len(data), max_length, vector_size), dtype=np.float32)
    for i,d in enumerate(data):
        for j,wd in enumerate(tokenizer.tokenize(d)):
            if j < max_length and wd in model.wv.vocab:
                zero_padding[i,j] = model[wd]

    zero_padding = np.expand_dims(zero_padding, axis=3)
    return zero_padding

    '''
    def query2vec_concat(q1,q2):
        zero_padding1 = np.zeros((len(q1), max_length , vector_size), dtype=np.float32)
        for i,q in enumerate(q1):
            idx = 0
            for wq in (q):
                if wq in model.wv.vocab:
                    if idx < max_length: # 0 ~ 49 is for query1 vertor
                        zero_padding1[i,idx] =  np.array(model[wq])
                        idx += 1


        zero_padding2 = np.zeros((len(q2), max_length , vector_size), dtype=np.float32)
        for i,q in enumerate(q2):
            idx = 0
            for wq in (q):
                if wq in model.wv.vocab:
                    if idx < max_length: # 50 ~ 99 is for query2 vertor
                        zero_padding2[i,idx] = np.array(model[wq])
                        idx += 1


        return zero_padding1, zero_padding2

    qvec1, qvec2 = query2vec_concat(query1,query2)
    del model
    print("Query vec :",qvec1.shape)
    qvec1_expand = np.expand_dims(qvec1,axis=3)
    qvec2_expand = np.expand_dims(qvec2,axis=3)
    print("Query vec expand :",qvec1_expand.shape)
    return qvec1_expand, qvec2_expand'''


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


#dataset = KinQueryDataset(LOCAL_DATASET_PATH,400)