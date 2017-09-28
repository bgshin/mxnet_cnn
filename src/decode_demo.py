from keras.layers import Conv1D, Average, Multiply
from keras.layers import Dense, AveragePooling1D, Input, Lambda
from keras.models import Model
from keras.preprocessing import sequence

from dataset import Timer
import os
from keras import backend as K
import gensim
import numpy as np
import pickle

class SentimentAnalysis():
    def __init__(self, w2v_dim=400, maxlen=60, w2v_path='../data/w2v/w2v-400-semevaltrndev.gnsm'):
        self.w2v_dim = w2v_dim
        self.maxlen = maxlen
        self.embedding, self.vocab = self.load_embedding(w2v_path)
        self.load_model()


    def load_embedding(self, w2v_path):
        print('Loading w2v...')
        emb_model = gensim.models.KeyedVectors.load(w2v_path, mmap='r')

        print('creating w2v mat...')
        word_index = emb_model.vocab
        embedding_matrix = np.zeros((len(word_index) + 1, 400), dtype=np.float32)
        for word, i in word_index.items():
            embedding_vector = emb_model[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i.index] = embedding_vector

        return embedding_matrix, emb_model.vocab

    def load_model(self, modelpath = './model/newbests17-400-v2-3'):
        filter_sizes = (1, 2, 3, 4, 5)
        num_filters = 80
        hidden_dims = 20

        def prediction_model(model_input, model_path):
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

                attention_size = self.maxlen - sz + 1

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
            z = Dense(hidden_dims, activation="relu")(z)
            model_output = Dense(3, activation="softmax")(z)

            model = Model(model_input, model_output)
            model.load_weights(model_path)
            model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

            return model

        def attention_model(model_input):
            att_list = []
            for sz in filter_sizes:
                conv = Conv1D(num_filters,
                              sz,
                              padding="valid",
                              activation="relu",
                              strides=1)(model_input)
                conv = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(conv)
                att = AveragePooling1D(pool_size=num_filters)(conv)
                att_list.append(att)

            model = Model(model_input, att_list)
            model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

            return model

        input_shape = (self.maxlen, self.w2v_dim)
        model_input = Input(shape=input_shape)
        self.p_model = prediction_model(model_input, modelpath)
        self.a_model = attention_model(model_input)

        for i in range(len(self.a_model.layers)):
            self.a_model.layers[i].set_weights(self.p_model.layers[i].get_weights())


    def preprocess_x(self, sentences):
        x_text = [line for line in sentences]
        sentence_len_list = [len(line) for line in sentences]
        x = []
        for s in x_text:
            one_doc = []
            for token in s.strip().split(" "):
                try:
                    one_doc.append(self.vocab[token].index)
                except:
                    one_doc.append(len(self.vocab))

            x.append(one_doc)

        x = np.array(x)
        x = sequence.pad_sequences(x, maxlen=self.maxlen)
        x = self.embedding[x]

        return x, sentence_len_list

    def decode(self, sentences):
        x, sentence_len_list = self.preprocess_x(sentences)
        y = self.p_model.predict(x, batch_size=2000, verbose=0)
        attention_matrix = self.a_model.predict(x, batch_size=2000, verbose=0)
        return y, attention_matrix



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    sentence = "I feel a little bit tired today , but I am really happy !"
    sentence_neg = "Although the rain stopped , I hate this thick cloud in the sky ."
    sentences1 = [sentence for i in range(1)]
    sentences10 = [sentence for i in range(10)]
    sentences100 = [sentence for i in range(100)]
    sentences1000 = [sentence for i in range(1000)]
    sentences10000 = [sentence for i in range(10000)]
    sentences100000 = [sentence for i in range(100000)]

    with Timer("init..."):
        sa = SentimentAnalysis()

    y, att = sa.decode([sentence, sentence_neg])
    with open('output.pkl', 'wb') as handle:
        pickle.dump(y, handle)
        pickle.dump(att, handle)

    exit()
    for i in [1, 10, 100, 1000, 10000, 100000]:
        varname = 'sentences%d' % i

        with Timer("decode %s..." % varname):
            y, att = sa.decode(eval(varname))




# [init...]
# Elapsed: 50.20814561843872
# 1/1 [==============================] - 1s
# 1/1 [==============================] - 0s
# [decode sentences1...]
# Elapsed: 2.263317346572876
# 10/10 [==============================] - 0s
# 10/10 [==============================] - 0s
# [decode sentences10...]
# Elapsed: 0.09254097938537598
# 100/100 [==============================] - 0s
# 100/100 [==============================] - 0s
# [decode sentences100...]
# Elapsed: 0.16536641120910645
# 1000/1000 [==============================] - 0s
# 1000/1000 [==============================] - 0s
# [decode sentences1000...]
# Elapsed: 0.2981994152069092
# 10000/10000 [==============================] - 1s
# 10000/10000 [==============================] - 0s
# [decode sentences10000...]
# Elapsed: 2.2783617973327637
# 100000/100000 [==============================] - 68s
# 100000/100000 [==============================] - 58s
# [decode sentences100000...]
# Elapsed: 146.7458312511444
