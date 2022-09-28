import os
import re
import pandas as pd
import numpy as np
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter
from sklearn.feature_extraction.text import TfidfVectorizer


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
        reg_txt = re.sub(r"[:;/+\.-]", '', reg_txt)
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

tv = TfidfVectorizer(min_df=0.05, max_df=0.5, sublinear_tf=True)
docterm_tv = tv.fit_transform(np.array(docterm))
docterm_tfidf = docterm_tv.toarray()
docterm_tfidf = pd.DataFrame(docterm_tfidf)
print(docterm_tfidf.head())

label = pd.DataFrame(label)
label = label.rename(columns={0: "label"})
docterm_df = pd.concat([docterm_tfidf, label], axis=1)
print(docterm_df.head())
