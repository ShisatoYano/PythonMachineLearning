import os
import re
import pandas as pd
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter


base_dir = "/workspaces/PythonMachineLearning/preprocess_sample_data/chap7/data/data/"
sub_dirs = ["it-life-hack", "movie-enter"]
docterm = []
label = []
tmp1 = []
tmp2 = ''

t = Tokenizer()
token_filters = [POSKeepFilter(["名詞"])]
a = Analyzer(char_filters=[], tokenizer=t, token_filters=token_filters)

for i, sub_dir in enumerate(sub_dirs):
    files = os.listdir(base_dir + sub_dir)

    for file in files:
        f = open(base_dir + sub_dir + '/' + file, 'r', encoding="utf-8")
        txt = f.read()

        reg_txt = re.sub(r"[0-9a-zA-Z]+", '', txt)
        reg_txt = re.sub(r"[:;/+\.~]", '', reg_txt)
        reg_txt = re.sub(r"[\s\n]", '', reg_txt)

        for token in a.analyze(reg_txt):
            tmp1.append(token.surface)
            tmp2 = ' '.join(tmp1)
        docterm.append(tmp2)
        tmp1 = []

        label.append(i)

        f.close()

print(pd.DataFrame(docterm).head())
print(docterm[0])
