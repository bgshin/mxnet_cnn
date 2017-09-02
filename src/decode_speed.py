import os
os.environ['KERAS_BACKEND']='mxnet'
# os.environ['KERAS_BACKEND']='tensorflow'

from keras.layers import Convolution1D
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding
from keras.layers import merge
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sst import load_all, Timer
import os
import argparse
import numpy as np
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf



def run(w2vdim, attempt, gpunum):
    filter_sizes = (2, 3, 4, 5)
    num_filters = 32
    dropout_prob = 0.8
    hidden_dims = 50
    maxlen = 60
    batch_size = 32
    epochs = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum

    if os.environ['KERAS_BACKEND']=='tensorflow':
        def get_session(gpu_fraction=1):
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                        allow_growth=True)
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        ktf.set_session(get_session())


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
        with Timer("load model weights..."):
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

    with Timer('decode'):
        print model.predict(x_trn, verbose=1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=400, choices=[50, 300, 400], type=int)
    parser.add_argument('-t', default=8, choices=range(10), type=int)
    parser.add_argument('-g', default="1", choices=["0", "1", "2", "3"], type=str)
    args = parser.parse_args()

    run(args.d, args.t, args.g)







