import tkinter as tk
import tkinter.filedialog as tkfd
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

root = tk.Tk()
root.withdraw()

# select csv file
csv_path = tkfd.askopenfilename(filetypes=[("CSV", "*.csv")])
bank_df_new = pd.read_csv(csv_path, sep=',')

# balancing imbalacned data
X = np.array(bank_df_new.drop('y', axis=1))
Y = np.array(bank_df_new[['y']])
print(np.sum(Y == 1), np.sum(Y == 0))
sampler = RandomUnderSampler(random_state=42)
X, Y = sampler.fit_resample(X, Y)
print(np.sum(Y == 1), np.sum(Y == 0))
