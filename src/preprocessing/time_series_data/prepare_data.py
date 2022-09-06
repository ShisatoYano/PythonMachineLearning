import pandas as pd

data_df = pd.read_csv("/workspaces/PythonMachineLearning/preprocess_sample_data/chap6/data/data/energydata.csv", sep=',')
print(data_df.head())

data_df = data_df.interpolate()

# diff in minute
data_df["date"] = pd.to_datetime(data_df["date"], format="%Y-%m-%d %H:%M:%S")
data_df["dif_min"] = data_df["date"].diff().dt.total_seconds() / 60
data_df["dif_min"] = data_df["dif_min"].fillna(0)
data_df["cum_min"] = data_df["dif_min"].cumsum()
print(data_df[["date", "cum_min"]])
