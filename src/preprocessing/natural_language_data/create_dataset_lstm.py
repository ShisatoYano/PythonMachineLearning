import os
import re
import itertools
import pandas as pd
import numpy as np
from collections import Counter
from janome.tokenizer import Tokenizer
from keras_preprocessing import sequence


base_dir = "/workspaces/PythonMachineLearning/preprocess_sample_data/chap7/data/data/"
sub_dirs = ["it-life-hack", "movie-enter"]

wakati = []
labels = []

t = Tokenizer(wakati=True)

for i, sub_dir in enumerate(sub_dirs):
    files = os.listdir(base_dir + sub_dir)

    for file in files:
        f = open(base_dir + sub_dir + '/' + file, 'r', encoding="utf-8")
        txt = f.read()

        reg_txt = re.sub(r"[0-9a-zA-Z]+", '', txt)
        reg_txt = re.sub(r"[:;/+\.-]", '', reg_txt)
        reg_txt = re.sub(r"[\s\n]", '', reg_txt)

        wakati.append(list(t.tokenize(reg_txt)))
        labels.append(i)
        f.close()
print(len(wakati))
print(wakati[0])
print(labels[0])

# descending sort
words_freq = Counter(itertools.chain(* wakati))
dic = []
for word_uniq in words_freq.most_common():
    dic.append(word_uniq[0])
print(pd.DataFrame(dic).head())

# add id
dic_inv = {}
for i, word_uniq in enumerate(dic, start=1):
    dic_inv.update({word_uniq: i})
print(dic_inv)

# id vector of each doc
wakati_id = [[dic_inv[word] for word in waka] for waka in wakati]
print(wakati_id[0])

# integrate length
wakati_id = sequence.pad_sequences(np.array(wakati_id), maxlen=3382, padding='post')
labels = np.array(labels)
print(wakati_id[0])
