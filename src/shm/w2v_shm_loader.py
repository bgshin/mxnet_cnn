import numpy as np
import sys
import signal
from src.dataset import load_embedding, load_sst_data, load_s17_data
from keras.preprocessing import sequence
from pyshm import SharedNPArray


SST_DATA_PATH = 'sst/'
SST_W2V_PATH = 'w2v-400-amazon.gnsm'
S17_DATA_PATH = 's17/'
S17_W2V_PATH = 'w2v-400-semevaltrndev.gnsm'


def set_share_memory(nparray, name):
    print name, nparray.shape, nparray.dtype
    sys.stdout.flush()
    shm_obj = SharedNPArray(shape=nparray.shape, dtype=nparray.dtype, tag=name)
    shm_obj.copyto(nparray)
    return shm_obj

maxlen = 60


def load_sst_from_file():
    dim = 400
    embedding, vocab = load_embedding(w2v_path=SST_W2V_PATH)
    shm_embedding = set_share_memory(embedding, 'sst_embedding_%d' % dim)

    (x_trn, y_trn) = load_sst_data(vocab, target='trn', pathbase='../../data/sst/')
    x_trn = sequence.pad_sequences(x_trn, maxlen=maxlen)
    shm_x_trn = set_share_memory(x_trn, 'sst_x_trn_%d' % dim)
    shm_y_trn = set_share_memory(y_trn, 'sst_y_trn_%d' % dim)

    (x_dev, y_dev) = load_sst_data(vocab, target='dev', pathbase='../../data/sst/')
    x_dev = sequence.pad_sequences(x_dev, maxlen=maxlen)
    shm_x_dev = set_share_memory(x_dev, 'sst_x_dev_%d' % dim)
    shm_y_dev = set_share_memory(y_dev, 'sst_y_dev_%d' % dim)

    (x_tst, y_tst) = load_sst_data(vocab, target='tst', pathbase='../../data/sst/')
    x_tst = sequence.pad_sequences(x_tst, maxlen=maxlen)
    shm_x_tst = set_share_memory(x_tst, 'sst_x_tst_%d' % dim)
    shm_y_tst = set_share_memory(y_tst, 'sst_y_tst_%d' % dim)

    return shm_embedding, shm_x_trn, shm_y_trn, shm_x_dev, shm_y_dev, shm_x_tst, shm_y_tst


def load_s17_from_file():
    dim = 400
    embedding, vocab = load_embedding(w2v_path=S17_W2V_PATH)
    shm_embedding = set_share_memory(embedding, 's17_embedding_%d' % dim)

    (x_trn, y_trn) = load_s17_data(vocab, target='trn', pathbase='../../data/s17/')
    x_trn = sequence.pad_sequences(x_trn, maxlen=maxlen)
    shm_x_trn = set_share_memory(x_trn, 's17_x_trn_%d' % dim)
    shm_y_trn = set_share_memory(y_trn, 's17_y_trn_%d' % dim)

    (x_dev, y_dev) = load_s17_data(vocab, target='dev', pathbase='../../data/s17/')
    x_dev = sequence.pad_sequences(x_dev, maxlen=maxlen)
    shm_x_dev = set_share_memory(x_dev, 's17_x_dev_%d' % dim)
    shm_y_dev = set_share_memory(y_dev, 's17_y_dev_%d' % dim)

    (x_tst, y_tst) = load_s17_data(vocab, target='tst', pathbase='../../data/s17/')
    x_tst = sequence.pad_sequences(x_tst, maxlen=maxlen)
    shm_x_tst = set_share_memory(x_tst, 's17_x_tst_%d' % dim)
    shm_y_tst = set_share_memory(y_tst, 's17_y_tst_%d' % dim)

    return shm_embedding, shm_x_trn, shm_y_trn, shm_x_dev, shm_y_dev, shm_x_tst, shm_y_tst


print 'loading for sst'
shm_embedding_sst, shm_x_trn_sst, shm_y_trn_sst, shm_x_dev_sst, shm_y_dev_sst, shm_x_tst_sst, shm_y_tst_sst = \
    load_sst_from_file()

print 'loading for s17'
shm_embedding_s17, shm_x_trn_s17, shm_y_trn_s17, shm_x_dev_s17, shm_y_dev_s17, shm_x_tst_s17, shm_y_tst_s17 = \
    load_s17_from_file()


print 'now sleep...'
sys.stdout.flush()


def signal_handler(signal, frame):
    print('now terminated...')
    sys.exit(0)

# use 'kill -2 ###' to delete all /dev/shm/*
signal.signal(signal.SIGINT, signal_handler)
signal.pause()
