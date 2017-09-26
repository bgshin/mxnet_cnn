from sklearn.metrics import confusion_matrix
import numpy as np
from keras.layers import Input, Multiply
from keras.models import Model
from keras import backend as K
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

y     = [0,0,1,1,2,2,2,1]
y_hat = [0,2,0,1,0,1,2,2]

y_hat_one_hot = np.zeros((len(y_hat), 3))
y_hat_one_hot[np.arange(len(y_hat)), y_hat] = 1

def into_categorical(y_true, y_pred, target='positive'):
    y_pred_category = K.argmax(y_pred, axis=-1)
    if target == 'positive':
        y_true_target = K.cast(y_true > 1, 'float32')
        y_pred_target = K.cast(y_pred_category > 1, 'float32')
    else: # 'negative'
        y_true_target = K.cast(y_true < 1, 'float32')
        y_pred_target = K.cast(y_pred_category < 1, 'float32')

    return y_true_target, y_pred_target


def precision(y_true, y_pred, target='positive'):
    y_true_target, y_pred_target = into_categorical(y_true, y_pred, target=target)

    true_target = K.sum(Multiply()([y_true_target, y_pred_target]))
    predicted_target = K.sum(y_pred_target)

    precision_target = true_target / (predicted_target + K.epsilon())
    return precision_target

def recall(y_true, y_pred, target='positive'):
    y_true_target, y_pred_target = into_categorical(y_true, y_pred, target=target)

    true_target = K.sum(Multiply()([y_true_target, y_pred_target]))
    possible_target = K.sum(y_true_target)

    recall_target = true_target / (possible_target + K.epsilon())
    return recall_target


def f1_target(y_true, y_pred, target='positive'):
    p = precision(y_true, y_pred, target=target)
    r = recall(y_true, y_pred, target=target)
    return 2 * p * r / (p + r + K.epsilon())


def f1_avg(y_true, y_pred):
    f1_pos = f1_target(y_true, y_pred, target='positive')
    f1_neg = f1_target(y_true, y_pred, target='negative')

    return (f1_pos + f1_neg) / 2


def precision_works(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    # y_true = tf.Print(y_true, [y_true], message="y_true: ", summarize=8)
    # y_pred = tf.Print(y_pred, [y_pred], message="y_pred: ", summarize=24)
    y_pred_category = K.argmax(y_pred, axis=-1)
    # y_pred_category = tf.Print(y_pred_category, [y_pred_category], message="y_pred_category: ", summarize=8)
    y_true_pos = K.cast(y_true > 1, 'float32')
    # y_true_pos = tf.Print(y_true_pos, [y_true_pos], message="y_true_pos: ", summarize=8)
    y_true_neg = K.cast(y_true < 1, 'float32')
    y_pred_pos = K.cast(y_pred_category > 1, 'float32')
    # y_pred_pos = tf.Print(y_pred_pos, [y_pred_pos], message="y_pred_pos: ", summarize=8)
    y_pred_neg = K.cast(y_pred_category < 1, 'float32')
    true_pos = K.sum(Multiply()([y_true_pos, y_pred_pos]))
    # true_pos = tf.Print(true_pos, [true_pos], message="true_pos: ", summarize=8)
    true_neg = Multiply()([y_true_neg, y_pred_neg])
    predicted_pos = K.sum(y_pred_pos)

    precision_pos = true_pos / (predicted_pos + K.epsilon())
    # return K.sum(true_positives2)
    return precision_pos
    # return K.sum(y_true_pos)
    # return K.sum(y_pred_pos)
    # return true_positives


def fmeasure(y_true, y_pred):
    # (True, pred)
    # NN, NO, NP
    # ON, OO, OP
    # PN, PO, PP

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    NN = float(cm[0][0])
    NO = float(cm[0][1])
    NP = float(cm[0][2])

    ON = float(cm[1][0])
    OO = float(cm[1][1])
    OP = float(cm[1][2])

    PN = float(cm[2][0])
    PO = float(cm[2][1])
    PP = float(cm[2][2])

    accuracy = float(format((NN + OO + PP) / (sum(sum(cm)) + np.finfo(float).eps), '.4f'))
    precision_negative = float(format(NN / (sum(cm[:, 0]) + np.finfo(float).eps), '.4f'))
    precision_positive = float(format(PP / (sum(cm[:, 2]) + np.finfo(float).eps), '.4f'))

    recall_negative = float(format(NN / (sum(cm[0, :]) + np.finfo(float).eps), '.4f'))
    recall_positive = float(format(PP / (sum(cm[2, :]) + np.finfo(float).eps), '.4f'))

    F1_negative = float(
        format(2 * precision_negative * recall_negative / (precision_negative + recall_negative + np.finfo(float).eps),
               '.4f'))
    F1_positive = float(
        format(2 * precision_positive * recall_positive / (precision_positive + recall_positive + np.finfo(float).eps),
               '.4f'))

    print('precision_negative=%f, recall_negative=%f, F1_negative=%f' % (
        precision_negative, recall_negative, F1_negative))
    print(
        'precision_positive=%f, recall_positive=%f, F1_positive=%f' % (
            precision_positive, recall_positive, F1_positive))

    F1 = (F1_negative+F1_positive)/2

    return accuracy, F1

accuracy, F1 = fmeasure(y, y_hat)

print('accuracy=%f, F1=%f' % (accuracy, F1))

input_shape = (3,)
model_input = Input(shape=input_shape)
model_output = model_input
model = Model(model_input, model_output)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=[precision, recall, f1_target, f1_avg])

model.fit(y_hat_one_hot, y,
          shuffle=False,
          batch_size=8,
          epochs=1)

y_pred = model.predict(y_hat_one_hot)
print(y_pred)

