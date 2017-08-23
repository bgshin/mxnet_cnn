import numpy as np
import gensim
from keras.preprocessing import sequence
from pyshm import SharedNPArray
import time
import re
import os


MAX_CHILDREN = 10
OUT_PATH = '../../'
DATA_PATH = 'cnntweets/data/rt-data-nlp4jtok/'
W2VBASE_PATH = 'cnntweets/data/emory_w2v/w2v-%d-amazon.gnsm'
# relation_index={'punct': 1,
#               'det': 2,
#               'prep': 3,
#               'nmod': 4,
#               'pobj': 5,
#               'nsubj': 6,
#               'advmod': 7,
#               'conj': 8,
#               'compound': 9,
#               'cc': 10,
#               'dobj': 11,
#               'aux': 12,
#               'poss': 13,
#               'acomp': 14,
#               'advcl': 15,
#               'relcl': 16,
#               'attr': 17,
#               'mark': 18,
#               'ccomp': 19,
#               'xcomp': 20,
#               'neg': 21,
#               'others':22,
#               None: 0,
#               'root': 0}

relation_index = {'punct': 1,
                  'det': 2,
                  'prep': 3,
                  'nmod': 4,
                  'pobj': 5,
                  'nsubj': 6,
                  'advmod': 7,
                  'conj': 8,
                  'compound': 9,
                  'cc': 10,
                  'dobj': 11,
                  'aux': 12,
                  'poss': 13,
                  'acomp': 14,
                  'advcl': 15,
                  'others': 16,
                  None: 0,
                  }

class ParsedSentence:
    def __init__(self, tsv):
        self.tsv = tsv
        self.n = len(tsv)

    def get_token(self, node):
        return self.tsv[node][1]

    def get_parent_id(self, node):
        parent_id = self.tsv[node][5]
        return int(parent_id) - 1

    def get_parant_token(self, node):
        id = self.get_parent_id(node)
        if id == -1:
            return 'ROOT'
        return self.get_token(id)

    def get_relationship(self, node):
        return self.tsv[node][6]


class TSVReader:
    def __init__(self, filename):
        self.ins = None
        self.empty = False
        self.open(filename)

    def __exit__(self):
        self.close()

    def open(self, filename):
        self.ins = open(filename)
        return self.ins

    def close(self):
        self.ins.close()

    def next_sentence(self):
        tsv = []

        self.empty = True
        for line in self.ins:
            self.empty = False
            line = line.strip()
            if line:
                tsv.append(re.compile('\t').split(line))
            elif tsv:
                return tsv
            else:
                return None


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


def to_treebase_data(target, vocab, pathbase='../data/%s.new.nlp', maxlen = 60, maxchildren = MAX_CHILDREN):
    filename = pathbase % target
    reader = TSVReader(filename)
    num = 0

    token_index_list = []
    children_list = []
    relation_list = []

    def voca_lookup(voca, token):
        if voca.has_key(token):
            index = vocab[token].index
        else:
            index = 0

        return index

    while reader.empty is False:
        tsv = reader.next_sentence()
        if tsv is None:
            break
        else:
            num += 1

        token_list = []
        token_index_one_sample_list = []
        parsed = ParsedSentence(tsv)
        children_dic = {}
        relation_name_dic = {}
        relation_dic = {}
        for i in range(parsed.n):
            current_token = parsed.get_token(i)
            token_list.append(current_token)

            current_voca_index = voca_lookup(vocab, current_token)
            token_index_one_sample_list.append(current_voca_index)
            parent_token = parsed.get_parant_token(i)
            relation = parsed.get_relationship(i)

            if relation not in relation_index:
                relation = 'others'

            if parent_token in children_dic:
                children_dic[parent_token].append(current_token)
                relation_name_dic[parent_token].append(relation)
                relation_dic[parent_token].append(relation_index[relation])

            else:
                children_dic[parent_token]=[current_token]
                relation_name_dic[parent_token]=[relation]
                relation_dic[parent_token]=[relation_index[relation]]

        children_one_sample_list = []
        relation_one_sample_list = []
        relation_name_list = []
        for tok in token_list:
            current_children = [0] * maxchildren
            current_rel = [0] * maxchildren
            current_rel_name = [None] * maxchildren

            if tok in children_dic:
                for idx, c in enumerate(children_dic[tok]):
                    current_voca_index = voca_lookup(vocab, c)
                    if idx >= MAX_CHILDREN:
                        continue

                    current_children[idx] = current_voca_index
                    current_rel_name[idx] = relation_name_dic[tok][idx]
                    current_rel[idx] = relation_dic[tok][idx]

                c = ' | '.join('{}({},{})'.format(*t) for t in zip(children_dic[tok], relation_name_dic[tok], relation_dic[tok]))

                # print '[%s]: %s' % (tok, c)

            # else:
            #     print '[%s]' % tok

            children_one_sample_list.append(current_children)
            relation_name_list.append(current_rel_name)
            relation_one_sample_list.append(current_rel)

        # print token_index_one_sample_list
        # print children_one_sample_list
        # print relation_one_sample_list

        padded_token_index_one_sample_list = [0] * (maxlen - len(token_index_one_sample_list)) + token_index_one_sample_list
        token_index_list.append(padded_token_index_one_sample_list)
        padded_children_one_sample_list = [[0] * maxchildren] * (maxlen - len(children_one_sample_list)) + children_one_sample_list
        children_list.append(padded_children_one_sample_list)
        padded_relation_one_sample_list = [[0] * maxchildren] * (maxlen - len(relation_one_sample_list)) + relation_one_sample_list
        relation_list.append(padded_relation_one_sample_list)

        # if num_example == num:
        #     break

    return np.array(token_index_list), np.array(children_list), np.array(relation_list)
    # return token_index_list, children_list, relation_list



def load_data(vocab, target='trn', pathbase='../data/'):
    x_text = [line.split('\t')[2] for line in open(pathbase + target, "r").readlines()]
    x = []
    for s in x_text:
        one_doc = []
        for token in s.strip().split(" "):
            try:
                one_doc.append(vocab[token].index)
            except:
                one_doc.append(len(vocab))

        x.append(one_doc)

    y = []
    for line in open(pathbase + target, "r").readlines():
        senti = line.split('\t')[1]
        if senti == 'neutral':
            y.append(2)
        elif senti == 'positive':
            y.append(3)
        elif senti == 'very_positive':
            y.append(4)
        elif senti == 'negative':
            y.append(1)
        elif senti == 'very_negative':
            y.append(0)

    return np.array(x), np.array(y)


def get_embedding(dim, base_path=OUT_PATH):
    print('Loading w2v...')
    if dim==300:
        W2VGSIM_DIR = 'cove/glove.840B.300d.w2vformat.gnsm'
        emb_model = gensim.models.KeyedVectors.load(base_path+W2VGSIM_DIR, mmap='r')
    else:
        emb_model = gensim.models.KeyedVectors.load( base_path+W2VBASE_PATH %(dim), mmap='r')

    print('creating w2v mat...')
    word_index = emb_model.vocab
    embedding_matrix = np.zeros((len(word_index) + 1, dim), dtype=np.float32)
    for word, i in word_index.items():
        embedding_vector = emb_model[word]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i.index] = embedding_vector

    return embedding_matrix, emb_model.vocab


def load_all(dim, maxlen, source='file'):
    if source=='shm':
        if dim == 50:
            max_features = 2665792
            embedding = SharedNPArray(shape=(max_features, 50), dtype=np.float32, tag='embedding_%d' % dim,
                                      create=False)

        elif dim == 300:
            max_features = 2196016
            embedding = SharedNPArray(shape=(max_features, 300), dtype=np.float32, tag='embedding_%d' % dim,
                                      create=False)

        else:  # dim==400
            max_features = 2665792
            embedding = SharedNPArray(shape=(max_features, 400), dtype=np.float32, tag='embedding_%d' % dim,
                                      create=False)

        x_trn = SharedNPArray(shape=(8544, 60), dtype=np.int32, tag='x_trn_%d' % dim, create=False)
        y_trn = SharedNPArray(shape=(8544,), dtype=np.int64, tag='y_trn_%d' % dim, create=False)
        x_dev = SharedNPArray(shape=(1101, 60), dtype=np.int32, tag='x_dev_%d' % dim, create=False)
        y_dev = SharedNPArray(shape=(1101,), dtype=np.int64, tag='y_dev_%d' % dim, create=False)
        x_tst = SharedNPArray(shape=(2210, 60), dtype=np.int32, tag='x_tst_%d' % dim, create=False)
        y_tst = SharedNPArray(shape=(2210,), dtype=np.int64, tag='y_tst_%d' % dim, create=False)


    else: # source=='file
        if dim == 50:
            max_features = 2665792

        elif dim == 300:
            max_features = 2196016

        else:  # dim==400
            max_features = 2665792

        embedding, vocab = get_embedding(dim)
        (x_trn, y_trn) = load_data(vocab, target='trn')
        (x_dev, y_dev) = load_data(vocab, target='dev')
        (x_tst, y_tst) = load_data(vocab, target='tst')
        x_trn = sequence.pad_sequences(x_trn, maxlen=maxlen)
        x_dev = sequence.pad_sequences(x_dev, maxlen=maxlen)
        x_tst = sequence.pad_sequences(x_tst, maxlen=maxlen)

    return (x_trn, y_trn), (x_dev, y_dev), (x_tst, y_tst), embedding, max_features




def load_tbdata(dim, source='file'):
    if source=='shm':
        if dim == 50:
            embedding = SharedNPArray(shape=(2665792, 50), dtype=np.float32, tag='tb_embedding_%d' % dim, create=False)

        else: # dim==400
            embedding = SharedNPArray(shape=(2665792, 400), dtype=np.float32, tag='tb_embedding_%d' % dim, create=False)

        tokens_etrn = SharedNPArray(shape=(165361, 60), dtype=np.int64, tag='tb_tokens_etrn_%d' % dim, create=False)
        children_etrn = SharedNPArray(shape=(165361, 60, MAX_CHILDREN), dtype=np.int64, tag='tb_children_etrn_%d' % dim,
                                      create=False)
        rels_etrn = SharedNPArray(shape=(165361, 60, MAX_CHILDREN), dtype=np.int64, tag='tb_relations_etrn_%d' % dim,
                                  create=False)
        y_etrn = SharedNPArray(shape=(165361,), dtype=np.int64, tag='tb_y_etrn_%d' % dim, create=False)

        tokens_trn = SharedNPArray(shape=(8544, 60), dtype=np.int64, tag='tb_tokens_trn_%d' % dim, create=False)
        children_trn = SharedNPArray(shape=(8544, 60, MAX_CHILDREN), dtype=np.int64, tag='tb_children_trn_%d' % dim,
                                     create=False)
        rels_trn = SharedNPArray(shape=(8544, 60, MAX_CHILDREN), dtype=np.int64, tag='tb_relations_trn_%d' % dim,
                                 create=False)
        y_trn = SharedNPArray(shape=(8544,), dtype=np.int64, tag='tb_y_trn_%d' % dim, create=False)

        tokens_dev = SharedNPArray(shape=(1101, 60), dtype=np.int64, tag='tb_tokens_dev_%d' % dim, create=False)
        children_dev = SharedNPArray(shape=(1101, 60, MAX_CHILDREN), dtype=np.int64, tag='tb_children_dev_%d' % dim, create=False)
        rels_dev = SharedNPArray(shape=(1101, 60, MAX_CHILDREN), dtype=np.int64, tag='tb_relations_dev_%d' % dim, create=False)
        y_dev = SharedNPArray(shape=(1101,), dtype=np.int64, tag='tb_y_dev_%d' % dim, create=False)

        tokens_tst = SharedNPArray(shape=(2210, 60), dtype=np.int64, tag='tb_tokens_tst_%d' % dim, create=False)
        children_tst = SharedNPArray(shape=(2210, 60, MAX_CHILDREN), dtype=np.int64, tag='tb_children_tst_%d' % dim, create=False)
        rels_tst = SharedNPArray(shape=(2210, 60, MAX_CHILDREN), dtype=np.int64, tag='tb_relations_tst_%d' % dim, create=False)
        y_tst = SharedNPArray(shape=(2210,), dtype=np.int64, tag='tb_y_tst_%d' % dim, create=False)
        max_features = 2665792


    else: # source=='file
        embedding, vocab = get_embedding(dim)
        tokens_etrn, children_etrn, rels_etrn = to_treebase_data('ext_trn', vocab, pathbase='../data/%s.new.nlp')
        (_, y_etrn) = load_data(vocab, target='ext_trn', pathbase='../data/')

        tokens_trn, children_trn, rels_trn = to_treebase_data('trn', vocab, pathbase='../data/%s.new.nlp')
        (_, y_trn) = load_data(vocab, target='trn', pathbase='../data/')

        tokens_dev, children_dev, rels_dev = to_treebase_data('dev', vocab, pathbase='../data/%s.new.nlp')
        (_, y_dev) = load_data(vocab, target='dev', pathbase='../data/')

        tokens_tst, children_tst, rels_tst = to_treebase_data('tst', vocab, pathbase='../data/%s.new.nlp')
        (_, y_tst) = load_data(vocab, target='tst', pathbase='../data/')

        max_features = 2665792

    return (tokens_etrn, children_etrn, rels_etrn, y_etrn), \
           (tokens_trn, children_trn, rels_trn, y_trn), \
           (tokens_dev, children_dev, rels_dev, y_dev), \
           (tokens_tst, children_tst, rels_tst, y_tst), embedding, max_features
