import tkinter as tk
import tkinter.filedialog as tkfd
import pandas as pd

root = tk.Tk()
root.withdraw()

# select csv file
csv_path = tkfd.askopenfilename(filetypes=[("CSV", "*.csv")])
bank_df = pd.read_csv(csv_path, sep=',')

print(bank_df.shape)
print(bank_df.dtypes)

# remove nan
bank_df = bank_df.dropna(subset=["job", "education"])
print(bank_df.shape)

# fill nan
print(bank_df.head())
bank_df = bank_df.fillna({"contact": "unknown"})
print(bank_df.head())

# remove outliers
bank_df = bank_df[bank_df["age"] >= 18]
bank_df = bank_df[bank_df["age"] < 100]
print(bank_df.shape)

# replace string to value
bank_df = bank_df.replace("yes", 1)
bank_df = bank_df.replace("no", 0)
print(bank_df.head())

# one-hot expression
bank_df_job = pd.get_dummies(bank_df["job"])
bank_df_marital = pd.get_dummies(bank_df["marital"])
bank_df_education = pd.get_dummies(bank_df["education"])
bank_df_contact = pd.get_dummies(bank_df["contact"])
bank_df_month = pd.get_dummies(bank_df["month"])

# create data set
# extract data which type is only value
tmp1 = bank_df[["age", "default", "balance", "housing", "loan",
                "day", "duration", "campaign", "pdays", "previous", "y"]]
print(tmp1)

# concatenate data frames
tmp2 = pd.concat([tmp1, bank_df_marital], axis=1)
tmp3 = pd.concat([tmp2, bank_df_education], axis=1)
tmp4 = pd.concat([tmp3, bank_df_contact], axis=1)
bank_df_new = pd.concat([tmp4, bank_df_month], axis=1)
print(bank_df_new)
