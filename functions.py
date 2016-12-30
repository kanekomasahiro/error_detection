import sys
import datetime
import collections
import pickle

vocab_num = 1

def fill_batch(batch, token='EOS'):
    max_len = max(len(x) for x in batch)
    return [x + [token] * (max_len - len(x)) for x in batch]

def take_len(train_txt):
    word_freq = collections.defaultdict(lambda: 0)
    count_l = []
    txt = []
    with open(train_txt) as f:
        for line in f:
            line_l = line.split('\t')
            words = line_l[-1].split()
            for word in words:
                word_freq[word] += 1
    for key, value in word_freq.items():
        if value > vocab_num:
            count_l.append(key)
    return len(count_l) + 3

def convert_word(sentence, word2id):
    return [word2id[word] for word in sentence]

def make_dict(train_txt, word2id, id2word, word_freq):
    word_l = []
    id2word = {}
    with open(train_txt) as f:
        for line in f:
            line_l = line.split('\t')
            words = line_l[-1].split()
            for word in words:
                word_freq[word] += 1
    for key, value in word_freq.items():
        if key not in word_l:
            if value > vocab_num:
                word_l.append(key)
                word2id[key] = len(word_l) + 1
                id2word[len(word_l) + 1] = key
    return word2id, id2word, word_l, word_freq
