# https://faroit.github.io/keras-docs/1.2.2/models/model/#methods
from keras.preprocessing import sequence
from keras.layers import Conv1D, TimeDistributed, Average, Multiply, Flatten
from keras.layers import Dense, Dropout, AveragePooling1D, Input, MaxPooling1D, Embedding, Lambda, Add
from keras.layers.merge import Concatenate
from keras.models import Model
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import ModelCheckpoint
from dataset import load_sst, load_s17, Timer
import os
from keras import backend as K
import argparse
import numpy as np
from keras.engine.topology import Layer

# class AttentionGram(Layer):
#
#     def __init__(self, att_dim, output_dim, **kwargs):
#         self.att_dim = att_dim
#         self.output_dim = output_dim
#         super(AttentionGram, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='attention_for_grams',
#                                       shape=(self.att_dim, ),
#                                       initializer='uniform',
#                                       trainable=True)
#         super(AttentionGram, self).build(input_shape)  # Be sure to call this somewhere!
#
#     def call(self, x_list):
#         x_weighted_list = []
#         for idx, x in enumerate(x_list):
#             x_weighted = Multiply()([x, self.kernel[idx]])
#             x_weighted_list.append(x_weighted)
#         result = Concatenate()(x_weighted_list)
#
#         return result
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)
#
#

def evaluation_summary(model, x_trn, y_trn,
                       x_dev, y_dev,
                       x_tst, y_tst, eval_name, trn_score=None, dev_score=None):


    score_list = []
    if trn_score is None:
        y_trn_student_softmax_prediction = model.predict(x_trn, verbose=0, batch_size=1000)
        y_trn_student_prediction = np.argmax(y_trn_student_softmax_prediction, axis=1)
        trn_score = sum(y_trn_student_prediction == y_trn) * 1.0 / len(y_trn)
        score_list.append(trn_score)
    else:
        score_list.append(trn_score)

    print ('trn score=%f' % trn_score)


    if dev_score is None:
        y_dev_student_softmax_prediction = model.predict(x_dev, verbose=0, batch_size=1000)
        y_dev_student_prediction = np.argmax(y_dev_student_softmax_prediction, axis=1)
        dev_score = sum(y_dev_student_prediction == y_dev) * 1.0 / len(y_dev)
        score_list.append(dev_score)
    else:
        score_list.append(dev_score)

    print ('dev score=%f' % dev_score)

    y_tst_student_softmax_prediction = model.predict(x_tst, verbose=0, batch_size=1000)
    y_tst_student_prediction = np.argmax(y_tst_student_softmax_prediction, axis=1)
    score = sum(y_tst_student_prediction == y_tst) * 1.0 / len(y_tst)
    print ('tst score=%f' % score)

    score_list.append(score)

    print('[summary] %s' % eval_name)
    print ('trn\tdev\ttst')
    print ('\t'.join(map(str, score_list)))

    return score_list


class MyCallback(ModelCheckpoint):
    def __init__(self, filepath, data, real_save=True, monitor='val_acc', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        print ('my callback init')
        super(MyCallback, self).__init__(filepath, monitor, verbose,
                                         save_best_only, save_weights_only,
                                         mode, period)

        self.x_trn, self.y_trn, self.x_dev, self.y_dev, self.x_tst, self.y_tst = data
        self.score_trn = 0
        self.score_dev = 0
        self.score_tst = 0
        self.real_save = real_save

    def evaluate(self, trn_score=None, dev_score=None):
        score_list = evaluation_summary(self.model, self.x_trn, self.y_trn, self.x_dev, self.y_dev,
                                        self.x_tst, self.y_tst, '[Epoch]', trn_score=trn_score, dev_score=dev_score)

        if self.score_dev < score_list[1]:
            self.score_trn = score_list[0]
            self.score_dev = score_list[1]
            self.score_tst = score_list[2]

        print('[Best]')
        print('\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst])))

    def on_train_end(self, logs=None):
        print('[Best:on_train_end]')
        print('\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst])))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.real_save == True:
            self.epochs_since_last_save += 1
            if self.epochs_since_last_save >= self.period:
                self.epochs_since_last_save = 0
                filepath = self.filepath.format(epoch=epoch, **logs)
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn('Can save best model only with %s available, '
                                      'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (epoch, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            self.evaluate()

                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve' %
                                      (epoch, self.monitor))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)

        else:
            current_val = logs.get(self.monitor)
            current_train = logs.get('acc')

            if self.monitor_op(current_val, self.best):
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f, trn(%0.5f)'
                          % (epoch, self.monitor, self.best,
                             current_val, current_train))
                self.best = current_val
                self.evaluate(trn_score=current_train, dev_score=current_val)

            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s did not improve' %
                          (epoch, self.monitor))


def run(attempt, gpunum, version):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum
    w2vdim = 400
    maxlen = 60
    batch_size = 200
    epochs = 100

    def CNNAttention_v1(model_input):
        print('model version V1')
        filter_sizes = (1, 2, 3, 4, 5)
        num_filters = 80
        dropout_prob = 0.2
        hidden_dims = 20


        conv_blocks = []
        for sz in filter_sizes:
            conv = Conv1D(num_filters,
                          sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model_input)
            print(conv)
            conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
            conv = AveragePooling1D(pool_size=num_filters)(conv)

            attention_size = maxlen - sz + 1

            multiplied_vector_list = []
            for i in range(attention_size):
                selected_attention = Lambda(lambda x: x[:, 0, i] / float(sz))(conv)

                for j in range(sz):
                    selected_token = Lambda(lambda x: x[:, i + j, :])(model_input)
                    multiplied_vector = Lambda(lambda x: Multiply()(x))([selected_token, selected_attention])

                    multiplied_vector_list.append(multiplied_vector)

            attentioned_conv = Average()(multiplied_vector_list)

            print(attentioned_conv)
            conv_blocks.append(attentioned_conv)

        z = Concatenate()(conv_blocks)
        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        z = Dropout(dropout_prob)(z)
        model_output = Dense(3, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

    def CNNAttention_v2(model_input):
        print('model version V2')
        filter_sizes = (1, 2, 3, 4, 5)
        num_filters = 80
        dropout_prob = 0.2
        hidden_dims = 20

        conv_blocks = []
        for sz in filter_sizes:
            conv = Conv1D(num_filters,
                          sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model_input)
            print(conv)
            conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
            conv = AveragePooling1D(pool_size=num_filters)(conv)

            attention_size = maxlen - sz + 1

            multiplied_vector_list = []
            for i in range(attention_size):
                selected_attention = Lambda(lambda x: x[:, 0, i] / float(sz))(conv)

                for j in range(sz):
                    selected_token = Lambda(lambda x: x[:, i + j, :])(model_input)
                    multiplied_vector = Lambda(lambda x: Multiply()(x))([selected_token, selected_attention])

                    multiplied_vector_list.append(multiplied_vector)

            attentioned_conv = Average()(multiplied_vector_list)

            print(attentioned_conv)
            conv_blocks.append(attentioned_conv)

        z = Average()(conv_blocks)
        # attgram = AttentionGram(len(filter_sizes), 400)
        # z = attgram(conv_blocks)
        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        z = Dropout(dropout_prob)(z)
        model_output = Dense(3, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model


    def CNN(model_input):
        print('Simple CNN V3')
        filter_sizes = (1, 2, 3, 4, 5)
        num_filters = 80
        dropout_prob = 0.8
        hidden_dims = 20

        conv_blocks = []
        for sz in filter_sizes:
            conv = Conv1D(num_filters,
                          sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model_input)
            print(conv)
            conv = MaxPooling1D(pool_size=2)(conv)
            print(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(dropout_prob)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(3, activation="softmax")(z)

        model = Model(model_input, model_output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model



    with Timer("load_all..."):
        (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features = \
            load_s17(w2vdim, maxlen, source='shm')



    with Timer("Build model..."):
        input_shape = (maxlen,w2vdim)
        model_input = Input(shape=input_shape)
        if version>0:
            model_class = eval('CNNAttention_v%d' % version)
        else:
            model_class = CNN
        model = model_class(model_input)
        model.summary()

    # checkpoint
    filepath='./model/newbests17-%d-%d' % (w2vdim, attempt)

    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]

    data = tuple((x_trn, y_trn,
                  x_dev, y_dev,
                  x_tst, y_tst
                  ))


    checkpoint = MyCallback(filepath, data, real_save=False, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='auto')
    callbacks_list = [checkpoint]

    model.fit(x_trn, y_trn,
              batch_size=batch_size,
              shuffle=True,
              callbacks=callbacks_list,
              epochs=epochs,
              validation_data=(x_dev, y_dev))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default=3, choices=range(10), type=int)
    parser.add_argument('-g', default="3", choices=["0", "1", "2", "3"], type=str)
    parser.add_argument('-v', default=0, choices=[0,1,2], type=int)
    args = parser.parse_args()

    run(args.t, args.g, args.v)







