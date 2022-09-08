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

# mean, std per 6 hours
data_df["cum_6hour"] = (data_df["cum_min"] / 360).round(2).astype(int)
print(data_df["cum_6hour"].unique())
print(data_df[["date", "cum_min", "cum_6hour"]].head(50))
data_df = data_df.drop(["date", "dif_min", "cum_min"], axis=1)
data_df_mean = data_df.groupby("cum_6hour").mean()
print(data_df_mean.head())
data_df_std = data_df.groupby("cum_6hour").std()
print(data_df_std.head())

# merge
data_features = pd.merge(data_df_mean, data_df_std, left_index=True, right_index=True)
print(data_features.head())
