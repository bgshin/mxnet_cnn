# https://faroit.github.io/keras-docs/1.2.2/models/model/#methods
import os
os.environ['KERAS_BACKEND']='mxnet'

from keras.layers import Convolution1D
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding
from keras.layers import merge
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sst import load_all, Timer
import os
import argparse

def run(w2vdim, attempt, gpunum):
    filter_sizes = (2, 3, 4, 5)
    num_filters = 32
    dropout_prob = 0.8
    hidden_dims = 50
    maxlen = 60
    batch_size = 32
    epochs = 30

    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum

    def CNNv1(model_input, max_features, model_path):
        z = Embedding(max_features,
                      w2vdim,
                      input_length=maxlen,
                      trainable=False)(model_input)

        conv_blocks = []
        for sz in filter_sizes:
            conv = Convolution1D(nb_filter=num_filters,
                                 filter_length=sz,
                                 border_mode="valid",
                                 activation="relu",
                                 subsample_length=1)(z)

            print(conv)
            conv = MaxPooling1D(pool_length=2)(conv)
            print(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        z = merge(conv_blocks, mode='concat')

        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(5, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.load_weights(model_path)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
                      context=["gpu(0)"])

        return model

    with Timer("load_all..."):
        (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features = \
            load_all(w2vdim, maxlen, source='shm')

    with Timer("Build model..."):
        input_shape = (maxlen,)
        model_input = Input(shape=input_shape)
        modelpath = './model/newbest-%d-%d' % (w2vdim, attempt)
        model = CNNv1(model_input, max_features, modelpath)
        model.summary()

    score_list = []
    score = model.evaluate(x_trn, y_trn, batch_size=4, verbose=1)
    print 'dev score=%f' % score[1]
    score_list.append(score[1])

    score = model.evaluate(x_dev, y_dev, batch_size=4, verbose=1)
    print 'dev score=%f' % score[1]
    score_list.append(score[1])

    score = model.evaluate(x_tst, y_tst, batch_size=4, verbose=1)
    print 'tst score=%f' % score[1]
    score_list.append(score[1])

    print '[summary]'
    print 'trn\tdev\ttst'
    print '\t'.join(map(str, score_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=400, choices=[50, 300, 400], type=int)
    parser.add_argument('-t', default=2, choices=range(10), type=int)
    parser.add_argument('-g', default="1", choices=["0", "1", "2", "3"], type=str)
    args = parser.parse_args()

    run(args.d, args.t, args.g)


