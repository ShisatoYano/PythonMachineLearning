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
bank_df = bank_df.dropna(thresh=2400, axis=1)
print(bank_df.shape)

# fill nan
print(bank_df.head())
bank_df = bank_df.fillna({"contact": "unknown"})
print(bank_df.head())

# remove outliers
bank_df = bank_df[bank_df["age"] >= 18]
bank_df = bank_df[bank_df["age"] < 100]
print(bank_df.shape)

# integrate string
bank_df.loc[(bank_df["job"] == "management") |
            (bank_df["job"] == "technician") |
            (bank_df["job"] == "blue-collar") |
            (bank_df["job"] == "admin") |
            (bank_df["job"] == "services") |
            (bank_df["job"] == "self-employed") |
            (bank_df["job"] == "entrepreneur") |
            (bank_df["job"] == "housemaid"), "job2"] = "worker"

bank_df.loc[(bank_df["job"] == "retired") |
            (bank_df["job"] == "unemployed") |
            (bank_df["job"] == "student"), "job2"] = "non-worker"

bank_df.loc[(bank_df["month"] == "jan") |
            (bank_df["month"] == "feb") |
            (bank_df["month"] == "mar"), "month2"] = "1Q"
bank_df.loc[(bank_df["month"] == "apr") |
            (bank_df["month"] == "may") |
            (bank_df["month"] == "jun"), "month2"] = "2Q"
bank_df.loc[(bank_df["month"] == "jul") |
            (bank_df["month"] == "aug") |
            (bank_df["month"] == "sep"), "month2"] = "3Q"
bank_df.loc[(bank_df["month"] == "oct") |
            (bank_df["month"] == "nov") |
            (bank_df["month"] == "dec"), "month2"] = "4Q"

# integrate value
bank_df.loc[bank_df["day"] <= 10, "day2"] = "early"
bank_df.loc[(bank_df["day"] > 10) & (bank_df["day"] <= 20), "day2"] = "middle"
bank_df.loc[bank_df["day"] > 20, "day2"] = "late"
bank_df.loc[bank_df["duration"] < 300, "duration2"] = "short"
bank_df.loc[bank_df["duration"] >= 300, "duration2"] = "long"
bank_df.loc[bank_df["previous"] < 1, "previous2"] = "zero"
bank_df.loc[bank_df["previous"] >= 1, "previous2"] = "one-more"
bank_df.loc[bank_df["pdays"] < 0, "pdays2"] = "less"
bank_df.loc[bank_df["pdays"] >= 0, "pdays2"] = "more"
print(bank_df)

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

# add more dummies
bank_df_job2 = pd.get_dummies(bank_df["job2"])
bank_df_month2 = pd.get_dummies(bank_df["month2"])
bank_df_day2 = pd.get_dummies(bank_df["day2"])
bank_df_duration2 = pd.get_dummies(bank_df["duration2"])
bank_df_previous2 = pd.get_dummies(bank_df["previous2"])
bank_df_pdays2 = pd.get_dummies(bank_df["pdays2"])
# concatenate
tmp5 = pd.concat([bank_df_new, bank_df_job2], axis=1)
tmp6 = pd.concat([tmp5, bank_df_month2], axis=1)
tmp7 = pd.concat([tmp6, bank_df_day2], axis=1)
tmp8 = pd.concat([tmp7, bank_df_duration2], axis=1)
tmp9 = pd.concat([tmp8, bank_df_previous2], axis=1)
bank_df_new2 = pd.concat([tmp9, bank_df_pdays2], axis=1)
print(bank_df_new2)
