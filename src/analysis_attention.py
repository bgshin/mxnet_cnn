import pickle
import numpy as np
import json

sentence = "I feel a little bit tired today , but I am really happy !"
sentence_neg = "Although the rain stopped , I hate this thick cloud in the sky ."

x_text = [line.split('\t')[2] for line in open('../data/s17/tst', "r").readlines()][:20]
x_text.append(sentence)
x_text.append(sentence_neg)

a= [[{"forms": ["This", "is", "the", "first", "document", "."], "offsets": [[0, 4], [5, 7], [8, 11], [12, 17], [18, 26], [26, 27]]}, {"forms": ["Contents", "of", "the", "first", "document", "are", "here", "."], "offsets": [[28, 36], [37, 39], [40, 43], [44, 49], [50, 58], [59, 62], [63, 67], [67, 68]]}], [{"forms": ["This", "is", "the", "second", "document", "."], "offsets": [[0, 4], [5, 7], [8, 11], [12, 18], [19, 27], [27, 28]]}, {"forms": ["The", "delimiter", "is", "not", "required", "for", "the", "last", "document", "."], "offsets": [[29, 32], [33, 42], [43, 45], [46, 49], [50, 58], [59, 62], [63, 66], [67, 71], [72, 80], [80, 81]]}]]

tokens = sentence.strip().split(' ')
tokens_neg = sentence_neg.strip().split(' ')
with open('output.pkl', 'rb') as handle:
    y = pickle.load(handle)
    att = pickle.load(handle)
    sentences = pickle.load(handle)

all_dic = []
for idx, s in enumerate(sentences):
    s_dic = {}
    s_dic['forms'] = s.split(' ')
    s_dic['sentiment'] = y[idx].tolist()
    for i in range(5):
        s_dic['sentiment-attention-%d' % (i+1)] = att[idx][i].tolist()

    all_dic.append(s_dic)

all_json = json.dumps(all_dic)
with open('samples.json', 'w') as outfile:
    json.dump(all_json, outfile)



for sent in sentences:
    print(sent)

i=0
gram_index = 0
sample_index =1
sentence_neg_len = len(tokens_neg)
sentence_len = len(tokens)
att[gram_index][sample_index][0]


all_att = []
for sample_index in range(2):
    one_sample_att = []
    for gram_index in range(5):
        norm_one_sample = att[gram_index][sample_index][0]/max(att[gram_index][sample_index][0])
        one_sample_att.append(norm_one_sample[-sentence_len+gram_index:])

    all_att.append(one_sample_att)


print('d')