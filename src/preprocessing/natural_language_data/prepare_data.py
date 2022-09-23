import os
import re
import pandas as pd
import numpy as np
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter
from sklearn.feature_extraction.text import CountVectorizer


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

# term-document matrix
cv = CountVectorizer()
docterm_cv = cv.fit_transform(np.array(docterm))
docterm_cnt = docterm_cv.toarray()
print(pd.DataFrame(docterm_cnt).head())

# descending order
word_count_pairs = []
docterm_wcnt = np.sum(a=docterm_cnt, axis=0)
for word, count in zip(cv.get_feature_names(), docterm_wcnt):
    word_count_pairs.append([word, count])
word_count_df = pd.DataFrame(word_count_pairs)
word_count_df = word_count_df.sort_values(1, ascending=False)
print(word_count_df.head())

# min-max
cv = CountVectorizer(min_df=0.01, max_df=0.5)
docterm_cv = cv.fit_transform(np.array(docterm))
docterm_cnt = docterm_cv.toarray()
print(pd.DataFrame(docterm_cnt).head())
