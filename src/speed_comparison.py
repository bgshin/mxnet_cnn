import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
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


def evaluation_summary(model, x_trn, y_trn,
                       x_dev, y_dev,
                       x_tst, y_tst, eval_name, trn_score=None, dev_score=None):


    score_list = []
    if trn_score is None:
        y_trn_student_softmax_prediction = model.predict(x_trn, verbose=0)
        y_trn_student_prediction = np.argmax(y_trn_student_softmax_prediction, axis=1)
        trn_score = sum(y_trn_student_prediction == y_trn) * 1.0 / len(y_trn)
        score_list.append(trn_score)
    else:
        score_list.append(trn_score)

    print 'trn score=%f' % trn_score

    if dev_score is None:
        y_dev_student_softmax_prediction = model.predict(x_dev, verbose=0, batch_size=10)
        y_dev_student_prediction = np.argmax(y_dev_student_softmax_prediction, axis=1)
        dev_score = sum(y_dev_student_prediction == y_dev) * 1.0 / len(y_dev)
        score_list.append(dev_score)
    else:
        score_list.append(dev_score)

    print 'dev score=%f' % dev_score

    y_tst_student_softmax_prediction = model.predict(x_tst, verbose=0, batch_size=10)
    y_tst_student_prediction = np.argmax(y_tst_student_softmax_prediction, axis=1)
    score = sum(y_tst_student_prediction == y_tst) * 1.0 / len(y_tst)
    print 'tst score=%f' % score
    score_list.append(score)

    print '[summary] %s' % eval_name
    print 'trn\tdev\ttst'
    print '\t'.join(map(str, score_list))

    return score_list

class MyCallback(ModelCheckpoint):
    def __init__(self, filepath, data, real_save=True, monitor='val_acc', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        print 'my callback init'
        super(MyCallback, self).__init__(filepath, monitor, verbose,
                                         save_best_only, save_weights_only,
                                         mode, period)

        self.x_trn, self.y_trn, self.x_dev, self.y_dev, self.x_tst, self.y_tst = data
        self.score_trn = 0
        self.score_dev = 0
        self.score_tst = 0
        self.real_save = real_save
        self.epoch_time_list = []
        self.save_time_list = []

    def evaluate(self, trn_score=None, dev_score=None):
        score_list = evaluation_summary(self.model, self.x_trn, self.y_trn, self.x_dev, self.y_dev,
                                        self.x_tst, self.y_tst, '[Epoch]', trn_score=trn_score, dev_score=dev_score)

        if self.score_dev < score_list[1]:
            self.score_trn = score_list[0]
            self.score_dev = score_list[1]
            self.score_tst = score_list[2]

        print '[Best]'
        print '\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst]))

    def on_train_end(self, logs=None):
        print '[Best:on_train_end]'
        print '\t'.join(map(str, [self.score_trn, self.score_dev, self.score_tst]))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

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

                            save_start = time.time()
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)

                            self.save_time = time.time() - save_start
                            self.save_time_list.append(self.save_time)
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
        self.epoch_time_list.append(time.time()-self.epoch_start - self.save_time)

def run(w2vdim, attempt, gpunum):
    filter_sizes = (2, 3, 4, 5)
    num_filters = 32
    dropout_prob = 0.8
    hidden_dims = 50
    maxlen = 60
    batch_size = 32
    epochs = 5

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

    with Timer("load_all..."):
        (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features = \
            load_all(w2vdim, maxlen, source='shm')

    with Timer("Build model..."):
        input_shape = (maxlen,)
        model_input = Input(shape=input_shape)
        model = CNNv1(model_input, max_features, embedding)
        model.summary()

    # checkpoint
    filepath='./model/newbest-%d-%d' % (w2vdim, attempt)
    data = tuple((x_trn, y_trn,
                  x_dev, y_dev,
                  x_tst, y_tst
                  ))

    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint = MyCallback(filepath, data, real_save=True, monitor='val_acc', verbose=1, save_best_only=True,
                            mode='auto')

    callbacks_list = [checkpoint]

    model.fit(x_trn, y_trn,
              batch_size=batch_size,
              shuffle=True,
              callbacks=callbacks_list,
              nb_epoch=epochs,
              validation_data=(x_dev, y_dev))

    print checkpoint.epoch_time_list
    print checkpoint.save_time_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=400, choices=[50, 300, 400], type=int)
    parser.add_argument('-t', default=9, choices=range(10), type=int)
    parser.add_argument('-g', default="1", choices=["0", "1", "2", "3"], type=str)
    args = parser.parse_args()

    run(args.d, args.t, args.g)







