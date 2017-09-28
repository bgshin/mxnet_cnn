import pickle

sentence = "I feel a little bit tired today , but I am really happy !"
sentence_neg = "Although the rain stopped , I hate this thick cloud in the sky ."

tokens = sentence.strip().split(' ')
tokens_neg = sentence_neg.strip().split(' ')
with open('output.pkl', 'rb') as handle:
    y = pickle.load(handle)
    att = pickle.load(handle)

i=0
gram_index = 0
sample_index =1
sentence_neg_len = len(tokens_neg)
att[gram_index][sample_index][0]
print('d')