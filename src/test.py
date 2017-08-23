# https://faroit.github.io/keras-docs/1.2.2/models/model/#methods
import os
os.environ['KERAS_BACKEND']='mxnet'
import keras as k

from keras.preprocessing import sequence
# from keras.layers.core import Dense, Dropout, Activation

from keras.layers import Convolution1D
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding
from keras.layers import merge
# from keras.layers.merge import Concatenate
import mxnet as mx
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sst import load_all, Timer
import os
import argparse
from keras import backend as K

def run(w2vdim, attempt, gpunum):
    filter_sizes = (2, 3, 4, 5)
    num_filters = 32
    dropout_prob = 0.8
    hidden_dims = 50
    maxlen = 60
    batch_size = 32
    epochs = 30

    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum
    os.environ["USE_CUDA"] = gpunum
    os.environ["USE_CUDA_PATH"] = '/usr/local/cuda'
    os.environ["USE_BLAS"] = 'blas'


    def CNNv1(model_input, max_features, embedding_matrix):
        z = Embedding(max_features,
                      w2vdim,
                      weights=[embedding_matrix],
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

        # z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        # Merge(layers=None, mode='sum', concat_axis=-1, dot_axes=-1, output_shape=None, output_mask=None, arguments=None,
        #       node_indices=None, tensor_indices=None, name=None)
        z = merge(conv_blocks, mode='concat')

        # mx.sym.Concat(a, b, dim=0)

        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(5, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model


    with Timer("load_all..."):
        (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features = \
            load_all(w2vdim, maxlen, source='shm')

    with Timer("Build model..."):
        input_shape = (maxlen,)
        model_input = Input(shape=input_shape)
        model = CNNv1(model_input, max_features, embedding)
        model.summary()
        # model = CNNv2(model_input, max_features)

    # checkpoint
    filepath='./model/newbest-%d-%d' % (w2vdim, attempt)

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(x_trn, y_trn,
              batch_size=batch_size,
              shuffle=True,
              callbacks=callbacks_list,
              nb_epoch=epochs,
              validation_data=(x_dev, y_dev))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=300, choices=[50, 300, 400], type=int)
    parser.add_argument('-t', default=1, choices=range(10), type=int)
    parser.add_argument('-g', default="0", choices=["0", "1", "2", "3"], type=str)
    args = parser.parse_args()

    run(args.d, args.t, args.g)







