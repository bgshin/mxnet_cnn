# https://faroit.github.io/keras-docs/1.2.2/models/model/#methods
import os
os.environ['KERAS_BACKEND']='mxnet'

from keras.layers import Convolution1D
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Embedding, Lambda, AveragePooling1D
from keras.layers import merge
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
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

        z = merge(conv_blocks, mode='concat')

        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(5, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
                      context=["gpu(0)"])

        return model


    def CNNAttention(model_input, max_features, embedding_matrix):
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

            conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
            # conv = MaxPooling1D(pool_size=num_filters)(conv)
            conv = AveragePooling1D(pool_length=num_filters)(conv)


            attention_size = maxlen - sz + 1

            multiplied_vector_list = []
            for i in range(attention_size):
                selected_attention = Lambda(lambda x: x[:,0,i]/float(sz))(conv)
                # selected_attention = Lambda(lambda x: x * float(sz))(selected_attention)

                for j in range(sz):
                    selected_token = Lambda(lambda x: x[:,i+j,:])(z)
                    multiplied_vector = Lambda(lambda x: merge(x, mode='mul'))([selected_token, selected_attention])

                    multiplied_vector_list.append(multiplied_vector)

            attentioned_conv = merge(multiplied_vector_list, mode='ave')
            # attentioned_conv = Average()(multiplied_vector_list)


            print(attentioned_conv)
            conv_blocks.append(attentioned_conv)

        z = merge(conv_blocks, mode='concat')
        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        z = Dropout(dropout_prob)(z)
        model_output = Dense(5, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    #
    # def CNNAttentionv2(model_input, max_features, embedding_matrix):
    #     z = Embedding(max_features,
    #                   w2vdim,
    #                   weights=[embedding_matrix],
    #                   input_length=maxlen,
    #                   trainable=False)(model_input)
    #
    #
    #     conv_blocks = []
    #     for sz in filter_sizes:
    #         conv = Conv1D(num_filters,
    #                       sz,
    #                       padding="valid",
    #                       activation="relu",
    #                       strides=1)(z)
    #         print(conv)
    #         conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
    #         # conv = MaxPooling1D(pool_size=num_filters)(conv)
    #         conv = AveragePooling1D(pool_size=num_filters)(conv)
    #
    #
    #         attention_size = maxlen - sz + 1
    #
    #         multiplied_vector_list = []
    #         for i in range(attention_size):
    #             selected_attention = Lambda(lambda x: x[:,0,i]/float(sz))(conv)
    #             # selected_attention = Lambda(lambda x: x * float(sz))(selected_attention)
    #
    #             for j in range(sz):
    #                 selected_token = Lambda(lambda x: x[:,i+j,:])(z)
    #                 multiplied_vector = Lambda(lambda x: Multiply()(x))([selected_token, selected_attention])
    #
    #                 multiplied_vector_list.append(multiplied_vector)
    #
    #         attentioned_conv = Average()(multiplied_vector_list)
    #
    #
    #         print(attentioned_conv)
    #         conv_blocks.append(attentioned_conv)
    #
    #     z = Add()(conv_blocks)
    #     z = Dropout(dropout_prob)(z)
    #     z = Dense(hidden_dims, activation="relu")(z)
    #     # z = Dropout(dropout_prob)(z)
    #     # z = Dense(50, activation="relu")(z)
    #     z = Dropout(dropout_prob)(z)
    #     model_output = Dense(5, activation="softmax")(z)
    #
    #     model = Model(model_input, model_output)
    #     model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #
    #     return model
    #
    #
    # def CNNAttentionv3(model_input, max_features, embedding_matrix):
    #     z = Embedding(max_features,
    #                   w2vdim,
    #                   weights=[embedding_matrix],
    #                   input_length=maxlen,
    #                   trainable=False)(model_input)
    #
    #
    #     conv_blocks = []
    #     for sz in filter_sizes:
    #         conv = Conv1D(num_filters,
    #                       sz,
    #                       padding="valid",
    #                       activation="relu",
    #                       strides=1)(z)
    #         print(conv)
    #         conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
    #         # conv = MaxPooling1D(pool_size=num_filters)(conv)
    #         conv = AveragePooling1D(pool_size=num_filters)(conv)
    #
    #
    #         attention_size = maxlen - sz + 1
    #
    #         multiplied_vector_list = []
    #         for i in range(attention_size):
    #             selected_attention = Lambda(lambda x: x[:,0,i]/float(sz))(conv)
    #             # selected_attention = Lambda(lambda x: x * float(sz))(selected_attention)
    #
    #             for j in range(sz):
    #                 selected_token = Lambda(lambda x: x[:,i+j,:])(z)
    #                 multiplied_vector = Lambda(lambda x: Multiply()(x))([selected_token, selected_attention])
    #
    #                 multiplied_vector_list.append(multiplied_vector)
    #
    #         attentioned_conv = Average()(multiplied_vector_list)
    #
    #
    #         print(attentioned_conv)
    #
    #         def expand_dims(x):
    #             return K.expand_dims(x, axis=1)
    #
    #         def expand_dims_output_shape(input_shape):
    #             return (None, 1, input_shape[1])
    #
    #         attentioned_conv_exp = Lambda(expand_dims, expand_dims_output_shape)(attentioned_conv)
    #         conv_blocks.append(attentioned_conv_exp)
    #
    #     z = Concatenate(axis=1)(conv_blocks)
    #     z = MaxPooling1D(pool_size=len(conv_blocks))(z)
    #     z = Lambda(lambda x: K.squeeze(x,1))(z)
    #     z = Dropout(dropout_prob)(z)
    #     z = Dense(50, activation="relu")(z)
    #     # z = Dropout(dropout_prob)(z)
    #     # z = Dense(50, activation="relu")(z)
    #     z = Dropout(dropout_prob)(z)
    #     model_output = Dense(5, activation="softmax")(z)
    #
    #     model = Model(model_input, model_output)
    #     model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #
    #     return model
    #
    #
    # def CNNAttentionv4(model_input, max_features, embedding_matrix):
    #     z = Embedding(max_features,
    #                   w2vdim,
    #                   weights=[embedding_matrix],
    #                   input_length=maxlen,
    #                   trainable=False)(model_input)
    #
    #
    #     conv_blocks = []
    #     for sz in filter_sizes:
    #         conv = Conv1D(num_filters,
    #                       sz,
    #                       padding="valid",
    #                       activation="relu",
    #                       strides=1)(z)
    #         print(conv)
    #         conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
    #         # conv = MaxPooling1D(pool_size=num_filters)(conv)
    #         conv = AveragePooling1D(pool_size=num_filters)(conv)
    #
    #
    #         attention_size = maxlen - sz + 1
    #
    #         multiplied_vector_list = []
    #         for i in range(attention_size):
    #             selected_attention = Lambda(lambda x: x[:,0,i]/float(sz))(conv)
    #             # selected_attention = Lambda(lambda x: x * float(sz))(selected_attention)
    #
    #             for j in range(sz):
    #                 selected_token = Lambda(lambda x: x[:,i+j,:])(z)
    #                 multiplied_vector = Lambda(lambda x: Multiply()(x))([selected_token, selected_attention])
    #
    #                 multiplied_vector_list.append(multiplied_vector)
    #
    #         attentioned_conv = Average()(multiplied_vector_list)
    #
    #
    #         print(attentioned_conv)
    #
    #         def expand_dims(x):
    #             return K.expand_dims(x, axis=1)
    #
    #         def expand_dims_output_shape(input_shape):
    #             return (None, 1, input_shape[1])
    #
    #         attentioned_conv_exp = Lambda(expand_dims, expand_dims_output_shape)(attentioned_conv)
    #         conv_blocks.append(attentioned_conv_exp)
    #
    #     z = Concatenate(axis=1)(conv_blocks)
    #     z = Lambda(lambda x: K.permute_dimensions(x,[0,2,1]))(z)
    #     z = TimeDistributed(Dense(1, activation="relu"))(z)
    #     z = Lambda(lambda x: K.squeeze(x,2))(z)
    #     z = Dropout(dropout_prob)(z)
    #     z = Dense(50, activation="relu")(z)
    #     z = Dropout(dropout_prob)(z)
    #     model_output = Dense(5, activation="softmax")(z)
    #
    #     model = Model(model_input, model_output)
    #     model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    #
    #     return model


    with Timer("load_all..."):
        (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features = \
            load_all(w2vdim, maxlen, source='shm')

    with Timer("Build model..."):
        input_shape = (maxlen,)
        model_input = Input(shape=input_shape)
        # model = CNNv1(model_input, max_features, embedding)
        model = CNNAttention(model_input, max_features, embedding)
        model.summary()

    # checkpoint
    filepath='./model/newbestatt-%d-%d' % (w2vdim, attempt)

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
    parser.add_argument('-d', default=400, choices=[50, 300, 400], type=int)
    parser.add_argument('-t', default=2, choices=range(10), type=int)
    parser.add_argument('-g', default="1", choices=["0", "1", "2", "3"], type=str)
    args = parser.parse_args()

    run(args.d, args.t, args.g)







