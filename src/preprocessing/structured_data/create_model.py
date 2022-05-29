import tkinter as tk
import tkinter.filedialog as tkfd
import pandas as pd
import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV

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
print(recall_score(Y_us[test_id], pred_y))
print(precision_score(Y_us[test_id], pred_y))

# optimize parameters by grid search
params = {
    "criterion": ["entropy"],
    "max_depth": [2, 4, 6, 8, 10],
    "min_samples_leaf": [10, 20, 30, 40, 50],
}
clf_gs = GridSearchCV(tree.DecisionTreeClassifier(),
                      params,
                      cv=KFold(n_splits=10, shuffle=True),
                      scoring="accuracy")
clf_gs.fit(X_us, Y_us)
print(clf_gs.best_score_)
print(clf_gs.best_params_)

# create model by using best parameters
clf_best = tree.DecisionTreeClassifier(
    criterion="entropy", max_depth=6, min_samples_leaf=20
)
clf_best.fit(X_us, Y_us)
print(clf_best.feature_importances_)
