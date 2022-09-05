import pandas as pd

data_df = pd.read_csv("/workspaces/PythonMachineLearning/preprocess_sample_data/chap6/data/data/energydata.csv", sep=',')
print(data_df.head())

# data's shape
print(data_df.shape)

# data's type
print(data_df.dtypes)

# convert to datetime
data_df["date"] = pd.to_datetime(data_df["date"], format="%Y-%m-%d %H:%M:%S")
print(data_df["date"].dtypes)
print(type(data_df["date"][0]))

# calculate epapsed time from initial time
data_df["dif_min"] = data_df["date"].diff().dt.total_seconds() / 60
data_df["dif_min"] = data_df["dif_min"].fillna(0)
print(data_df["dif_min"].head())
data_df["cum_min"] = data_df["dif_min"].cumsum()
print(data_df[["date", "cum_min"]].head())

# minute to hour
data_df["cum_hour"] = (data_df["cum_min"] / 60).round(2).astype(int)
print(data_df[["date", "cum_min", "cum_hour"]].head(10))

# mean, std by 1 hour
print(data_df.groupby("cum_hour").mean())
print(data_df.groupby("cum_hour").std())

# calculate statistics
print(data_df.describe())

# find Nan
print(data_df.isnull().sum(axis=1).sort_values(ascending=False))
print(data_df.isnull().sum(axis=0))
