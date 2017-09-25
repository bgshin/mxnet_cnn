import numpy as np
import gensim
from keras.preprocessing import sequence
from pyshm import SharedNPArray
import time
import re
import os


SST_DATA_PATH = 'sst/'
SST_W2V_PATH = 'w2v-400-amazon.gnsm'
S17_DATA_PATH = 's17/'
S17_W2V_PATH = 'w2v-400-semevaltrndev.gnsm'




class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
        print('Elapsed: %s' % (time.time() - self.tstart))


def load_s17_data(vocab, target='trn', pathbase='../data/s17/'):
    x_text = [line.split('\t')[2] for line in open(pathbase + target, "r").readlines()]
    x = []
    for s in x_text:
        one_doc = []
        for token in s.strip().split(" "):
            try:
                one_doc.append(vocab[token].index)
            except:
                one_doc.append(len(vocab))

        x.append(one_doc)

    y = []
    for line in open(pathbase + target, "r").readlines():
        senti = line.split('\t')[1]
        if senti == 'negative':
            y.append(0)
        elif senti == 'objective':
            y.append(1)
        elif senti == 'positive':
            y.append(2)

    return np.array(x), np.array(y)


def load_sst_data(vocab, target='trn', pathbase='../data/sst/'):
    x_text = [line.split('\t')[2] for line in open(pathbase + target, "r").readlines()]
    x = []
    for s in x_text:
        one_doc = []
        for token in s.strip().split(" "):
            try:
                one_doc.append(vocab[token].index)
            except:
                one_doc.append(len(vocab))

        x.append(one_doc)

    y = []
    for line in open(pathbase + target, "r").readlines():
        senti = line.split('\t')[1]
        if senti == 'neutral':
            y.append(2)
        elif senti == 'positive':
            y.append(3)
        elif senti == 'very_positive':
            y.append(4)
        elif senti == 'negative':
            y.append(1)
        elif senti == 'very_negative':
            y.append(0)

    return np.array(x), np.array(y)


def load_embedding(w2v_path=SST_W2V_PATH, pathbase='../data/w2v/'):
    print('Loading w2v...')
    emb_model = gensim.models.KeyedVectors.load( pathbase+w2v_path, mmap='r')

    print('creating w2v mat...')
    word_index = emb_model.vocab
    embedding_matrix = np.zeros((len(word_index) + 1, 400), dtype=np.float32)
    for word, i in word_index.items():
        embedding_vector = emb_model[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i.index] = embedding_vector

    return embedding_matrix, emb_model.vocab


def load_sst(dim, maxlen, source='file'):
    if source=='shm':
        max_features = 2665792
        embedding = SharedNPArray(shape=(max_features, 400), dtype=np.float32, tag='sst_embedding_%d' % dim,
                                  create=False)

        x_trn = SharedNPArray(shape=(8544, 60), dtype=np.int32, tag='sst_x_trn_%d' % dim, create=False)
        y_trn = SharedNPArray(shape=(8544,), dtype=np.int64, tag='sst_y_trn_%d' % dim, create=False)
        x_dev = SharedNPArray(shape=(1101, 60), dtype=np.int32, tag='sst_x_dev_%d' % dim, create=False)
        y_dev = SharedNPArray(shape=(1101,), dtype=np.int64, tag='sst_y_dev_%d' % dim, create=False)
        x_tst = SharedNPArray(shape=(2210, 60), dtype=np.int32, tag='sst_x_tst_%d' % dim, create=False)
        y_tst = SharedNPArray(shape=(2210,), dtype=np.int64, tag='sst_y_tst_%d' % dim, create=False)


    else: # source=='file
        max_features = 2665792

        embedding, vocab = load_embedding(w2v_path=SST_W2V_PATH)
        (x_trn, y_trn) = load_sst_data(vocab, target='trn')
        (x_dev, y_dev) = load_sst_data(vocab, target='dev')
        (x_tst, y_tst) = load_sst_data(vocab, target='tst')
        x_trn = sequence.pad_sequences(x_trn, maxlen=maxlen)
        x_dev = sequence.pad_sequences(x_dev, maxlen=maxlen)
        x_tst = sequence.pad_sequences(x_tst, maxlen=maxlen)

    return (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features


def load_s17(dim, maxlen, source='file'):
    if source=='shm':
        max_features = 3676787
        embedding = SharedNPArray(shape=(max_features, 400), dtype=np.float32, tag='s17_embedding_%d' % dim,
                                  create=False)
        embedding = embedding._SharedNPArray__np_array

        x_trn = SharedNPArray(shape=(15385, 60), dtype=np.int32, tag='s17_x_trn_%d' % dim, create=False)
        y_trn = SharedNPArray(shape=(15385,), dtype=np.int64, tag='s17_y_trn_%d' % dim, create=False)
        x_dev = SharedNPArray(shape=(1588, 60), dtype=np.int32, tag='s17_x_dev_%d' % dim, create=False)
        y_dev = SharedNPArray(shape=(1588,), dtype=np.int64, tag='s17_y_dev_%d' % dim, create=False)
        x_tst = SharedNPArray(shape=(20632, 60), dtype=np.int32, tag='s17_x_tst_%d' % dim, create=False)
        y_tst = SharedNPArray(shape=(20632,), dtype=np.int64, tag='s17_y_tst_%d' % dim, create=False)

        x_trn = x_trn._SharedNPArray__np_array
        y_trn = y_trn._SharedNPArray__np_array

        x_dev = x_dev._SharedNPArray__np_array
        y_dev = y_dev._SharedNPArray__np_array

        x_tst = x_tst._SharedNPArray__np_array
        y_tst = y_tst._SharedNPArray__np_array

        x_newdev = np.concatenate((x_trn[11880:], x_dev))
        y_newdev = np.concatenate((y_trn[11880:], y_dev))

        x_newtrn = x_trn[:11880]
        y_newtrn = y_trn[:11880]

        x_trn = embedding[x_newtrn]
        x_dev = embedding[x_newdev]
        # x_trn = embedding[x_trn]
        # x_dev = embedding[x_dev]
        x_tst = embedding[x_tst]

        y_trn = y_newtrn
        y_dev = y_newdev



    else: # source=='file
        max_features = 3676787

        embedding, vocab = load_embedding(w2v_path=S17_W2V_PATH)
        (x_trn, y_trn) = load_s17_data(vocab, target='trn')
        (x_dev, y_dev) = load_s17_data(vocab, target='dev')
        (x_tst, y_tst) = load_s17_data(vocab, target='tst')
        x_trn = sequence.pad_sequences(x_trn, maxlen=maxlen)
        x_dev = sequence.pad_sequences(x_dev, maxlen=maxlen)
        x_tst = sequence.pad_sequences(x_tst, maxlen=maxlen)

    return (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features
