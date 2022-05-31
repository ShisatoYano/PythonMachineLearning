import tkinter as tk
import tkinter.filedialog as tkfd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

root = tk.Tk()
root.withdraw()

# select csv file
csv_path = tkfd.askopenfilename(filetypes=[("CSV", "*.csv")])
bank_df = pd.read_csv(csv_path, sep=',')
print(bank_df.shape)
print(bank_df.dtypes)

# normalize data
# min-max normalization
bank_df = bank_df.drop("y", axis=1)
mc = MinMaxScaler()
mc.fit(bank_df)
bank_df_mc = pd.DataFrame(mc.transform(bank_df), columns=bank_df.columns)
print(bank_df_mc)

# z-score normalization
sc = StandardScaler()
sc.fit(bank_df)
bank_df_sc = pd.DataFrame(sc.transform(bank_df), columns=bank_df.columns)
print(bank_df_sc)
