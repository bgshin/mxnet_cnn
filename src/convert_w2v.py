# python -m gensim.scripts.glove2word2vec --input glove.840B.300d.txt --output glove.840B.300d.w2vformat.txt

import gensim
import time


W2V_DIR = '../data/w2v/w2v-400-semevaltrndev.bin'
W2VGSIM_DIR = '../data/w2v/w2v-400-semevaltrndev.gnsm'

print W2V_DIR
start_time = time.time()
# model = gensim.models.KeyedVectors.load_word2vec_format(W2V_DIR, binary=False)
model = gensim.models.KeyedVectors.load_word2vec_format(W2V_DIR, binary=True)
elapsed_time = time.time() - start_time
print(elapsed_time) # about 18 secs on my old PC
print('Found %s word vectors.' % len(model.vocab))
print model['apple']
print model.vocab['apple']
print model.syn0[4978]

#
# model.init_sims(replace=True)
print W2VGSIM_DIR
start_time = time.time()
model.save(W2VGSIM_DIR) ## saves a memory mapped npy file with it
elapsed_time = time.time() - start_time
print(elapsed_time) # about 30 secs on my old PC



# start_time = time.time()
# model = gensim.models.KeyedVectors.load(W2VGSIM_DIR, mmap='r')
# model.syn0norm = model.syn0
# elapsed_time = time.time() - start_time
# print(elapsed_time) # about 18 secs on my old PC
# print('Found %s word vectors.' % len(model.vocab))
# print model.syn0[4978]
# # Semaphore(0).acquire()




# start_time = time.time()
# model = gensim.models.KeyedVectors.load(W2VGSIM_DIR, mmap='r')
# model.syn0norm = model.syn0
# elapsed_time = time.time() - start_time
# print(elapsed_time) # about 18 secs on my old PC
# print('Found %s word vectors.' % len(model.vocab))
# print model['apple']


# def load_w2v():
#
#     w2v = 'w2v-50-amazon.bin'
#     model_path = W2V_DIR+w2v
#
#     model = Word2Vec.load_word2vec_format(model_path, binary=True)
#     print("The vocabulary size is: " + str(len(model.vocab)))
#
#     return model

