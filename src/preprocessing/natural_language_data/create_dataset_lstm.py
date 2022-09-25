import os
import re
from janome.tokenizer import Tokenizer


base_dir = "/workspaces/PythonMachineLearning/preprocess_sample_data/chap7/data/data/"
sub_dirs = ["it-life-hack", "movie-enter"]

words = []
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

        words.append(t.tokenize(reg_txt))
        labels.append(i)
        f.close()
print(len(words))
print(words[0])
print(labels[0])
