import tkinter as tk
import tkinter.filedialog as tkfd
import pandas as pd

root = tk.Tk()
root.withdraw()

# select csv file
csv_path = tkfd.askopenfilename(filetypes=[("CSV", "*.csv")])
bank_df = pd.read_csv(csv_path, sep=',')

# show data from 1st line till 5th line
print(bank_df.head())

# show dataframe size
print(bank_df.shape)

# show data type of each item, memory usage
print(bank_df.info())

# check dataframe include null
print(bank_df.isnull().any(axis=1))  # row direction
print(bank_df.isnull().any(axis=0))  # column direction
