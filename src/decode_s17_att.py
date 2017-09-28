from keras.layers import Conv1D, Average, Multiply
from keras.layers import Dense, Dropout, AveragePooling1D, Input, Lambda
from keras.models import Model
from dataset import load_sst, load_s17, Timer
import os
from keras import backend as K
import argparse



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


def f1(y_true, y_pred):
    f1_pos = f1_target(y_true, y_pred, target='positive')
    f1_neg = f1_target(y_true, y_pred, target='negative')

    return (f1_pos + f1_neg) / 2


def run(gpunum, version, attempt=3):
    w2vdim = 400
    maxlen = 60

    os.environ["CUDA_VISIBLE_DEVICES"] = gpunum


    def CNNAttention_v2(model_input, model_path):
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
            # conv = MaxPooling1D(pool_size=num_filters)(conv)
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

        # model = Model(model_input, model_output)
        # model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1])

        model = Model(model_input, model_output)
        model.load_weights(model_path)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1])

        return model


    def extract_attention(model_input):
        print('model version V2')
        filter_sizes = (1, 2, 3, 4, 5)
        num_filters = 80

        att_list = []
        for sz in filter_sizes:
            conv = Conv1D(num_filters,
                          sz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model_input)
            print(conv)
            conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
            att = AveragePooling1D(pool_size=num_filters)(conv)
            att_list.append(att)

        model = Model(model_input, att_list)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1])

        return model



    # with Timer("load_all..."):
    #     (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features = \
    #         load_s17(w2vdim, maxlen, source='shm')

    with Timer("load_all..."):
        (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features = \
            load_s17(w2vdim, maxlen, source='shm', new_split=False)

    with Timer("Build model..."):
        modelpath = './model/newbests17-%d-v%d-%d' % (w2vdim, version, attempt)


        input_shape = (maxlen, w2vdim)
        model_input = Input(shape=input_shape)
        if version > 0:
            model_class = eval('CNNAttention_v%d' % version)
        else:
            model_class = CNNAttention_v2
        model = model_class(model_input, modelpath)
        model.summary()

        att_model = extract_attention(model_input)

        for i in range(len(att_model.layers)):
            att_model.layers[i].set_weights(model.layers[i].get_weights())

    score_list = []
    y_trn_hat = model.predict(x_trn, batch_size=2000, verbose=1)
    print(y_trn_hat)
    # print('dev score=%f' % score[1])
    # score_list.append(score[1])

    print(model.metrics_names)
    score = model.evaluate(x_dev, y_dev, batch_size=2000, verbose=1)
    print('dev score=%f' % score[1])
    score_list.append(score[1])

    attention_matrix = att_model.predict(x_dev, batch_size=2000, verbose=1)

    score = model.evaluate(x_tst, y_tst, batch_size=2000, verbose=1)
    print('tst score=%f' % score[1])
    score_list.append(score[1])

    print('[summary]')
    print('trn\tdev\ttst')
    print('\t'.join(map(str, score_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default=3, choices=range(10), type=int)
    parser.add_argument('-g', default="3", choices=["0", "1", "2", "3"], type=str)
    parser.add_argument('-v', default=2, choices=[0, 1, 2, 3], type=int)  # 0:cnn, 1:att1, 2:att2
    args = parser.parse_args()

    run(args.g, args.v, args.t)


