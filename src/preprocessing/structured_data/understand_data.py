import tkinter as tk
import tkinter.filedialog as tkfd
import pandas as pd
import matplotlib.pyplot as plt

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

# check how many null is included in dataframe
print(bank_df.isnull().sum(axis=1))  # row direction
print(bank_df.isnull().sum(axis=0))  # column direction

# calculate statistical value of each item
print(bank_df.describe())

# visualize data
# histogram
# plt.hist(bank_df["age"])
# plt.xlabel("age")
# plt.ylabel("freq")

# scatter
# plt.scatter(bank_df["age"], bank_df["balance"])
# plt.xlabel("age")
# plt.ylabel("balance")
# calculate correlation coefficient
print(bank_df[["age", "balance"]].corr())

# pie chart
# count number of value
print(bank_df["job"].value_counts(ascending=False, normalize=True))
job_label = bank_df["job"].value_counts(ascending=False, normalize=True).index
print(job_label)
job_value = bank_df["job"].value_counts(ascending=False, normalize=True).values
print(job_value)
plt.pie(job_value, labels=job_label)
plt.axis("equal")

plt.show()
