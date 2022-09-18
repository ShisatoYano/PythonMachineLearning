from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import POSKeepFilter
import re
import collections

# morphological analysis sample by Janome
f = open("/workspaces/PythonMachineLearning/preprocess_sample_data/chap7/data/data/it-life-hack/it-life-hack-6292880.txt", encoding="utf-8")
txt = f.read()
print(txt)

# regular expression
reg_txt = re.sub(r"[0-9a-zA-Z]+", '', txt)
reg_txt = re.sub(r"[:;/+\.~]", '', reg_txt)
reg_txt = re.sub(r"[\s\n]", '', reg_txt)
print(reg_txt)

# analyze
t = Tokenizer()
token_filters = [POSKeepFilter(['名詞'])]
a = Analyzer(char_filters=[], tokenizer=t, token_filters=token_filters)
words_list = []
for token in a.analyze(reg_txt):
    words_list.append(token.surface)
# count words
c = collections.Counter(words_list)
print(c)

f.close()
