import tkinter as tk
import tkinter.filedialog as tkfd
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score

root = tk.Tk()
root.withdraw()

# select csv file
csv_path = tkfd.askopenfilename(filetypes=[("CSV", "*.csv")])
bank_df_new = pd.read_csv(csv_path, sep=',')

# balancing imbalacned data
X = np.array(bank_df_new.drop('y', axis=1))
Y = np.array(bank_df_new[['y']])
print(np.sum(Y == 1), np.sum(Y == 0))

# under sampling
sampler = RandomUnderSampler(random_state=42)
X_us, Y_us = sampler.fit_resample(X, Y)
print(np.sum(Y_us == 1), np.sum(Y_us == 0))

# over sampling
sampler = RandomOverSampler(random_state=42)
X_os, Y_os = sampler.fit_resample(X, Y)
print(np.sum(Y_os == 1), np.sum(Y_os == 0))

# create decision tree model
kf = KFold(n_splits=10, shuffle=True)
scores = []
# cross validation
for train_id, test_id in kf.split(X_us):
    x = X_us[train_id]
    y = Y_us[train_id]
    clf = tree.DecisionTreeClassifier()
    clf.fit(x, y)
    pred_y = clf.predict(X_us[test_id])
    score = accuracy_score(Y_us[test_id], pred_y)
    scores.append(score)
# calculate score
scores = np.array(scores)
print(scores.mean(), scores.std())
