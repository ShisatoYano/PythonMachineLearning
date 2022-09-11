import pandas as pd
import datetime as dt
import numpy as np

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

# objective variable
event_df = pd.read_csv("/workspaces/PythonMachineLearning/preprocess_sample_data/chap6/data/data/event.csv", sep=',')
print(event_df.head())
# diff min with base time
event_df["date"] = pd.to_datetime(event_df["date"], format="%Y-%m-%d %H:%M:%S")
base_time = "2016-01-11 17:00:00"
event_df["dif_min"] = event_df["date"] - dt.datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
event_df["dif_min"] = event_df["dif_min"].dt.total_seconds() / 60
event_df["cum_6hour"] = (event_df["dif_min"] / 360).round(2).astype(int)
print(event_df.head())
event_df["event"] = 1
event_df = event_df[["cum_6hour", "event"]]
event_df = event_df[~event_df.duplicated()]
event_df = event_df.set_index(["cum_6hour"])
print(event_df.head())
# join
data_event = data_features.join(event_df, how="left")
data_event = data_event.fillna(0)
print(data_event.head())

# slide window
hid = data_df["cum_6hour"].unique()
data_slide_features = []
for i in range(len(hid)):
    tmp = data_df[data_df["cum_6hour"] == i]
    tmp = tmp.drop(["cum_6hour"], axis=1)

    tmp_mean = tmp.rolling(6).mean()
    tmp_mean = tmp_mean.dropna()

    tmp_std = tmp.rolling(6).std()
    tmp_std = tmp_std.dropna()

    tmp2 = (np.array(tmp_mean["Appliances"]).tolist() + np.array(tmp_mean["lights"]).tolist() +
            np.array(tmp_mean["T1"]).tolist() + np.array(tmp_mean["RH_1"]).tolist() +
            np.array(tmp_mean["T2"]).tolist() + np.array(tmp_mean["RH_2"]).tolist() +
            np.array(tmp_mean["T3"]).tolist() + np.array(tmp_mean["RH_3"]).tolist() +
            np.array(tmp_mean["T4"]).tolist() + np.array(tmp_mean["RH_4"]).tolist() +
            np.array(tmp_mean["T5"]).tolist() + np.array(tmp_mean["RH_5"]).tolist() +
            np.array(tmp_std["Appliances"]).tolist() + np.array(tmp_std["lights"]).tolist() +
            np.array(tmp_std["T1"]).tolist() + np.array(tmp_std["RH_1"]).tolist() +
            np.array(tmp_std["T2"]).tolist() + np.array(tmp_std["RH_2"]).tolist() +
            np.array(tmp_std["T3"]).tolist() + np.array(tmp_std["RH_3"]).tolist() +
            np.array(tmp_std["T4"]).tolist() + np.array(tmp_std["RH_4"]).tolist() +
            np.array(tmp_std["T5"]).tolist() + np.array(tmp_std["RH_5"]).tolist())

    data_slide_features.append(tmp2)

data_slide_features = pd.DataFrame(data_slide_features)
print(data_slide_features.head())

print(data_slide_features.shape)
print(data_event.shape)
tmp3 = data_event[["event"]]
data_event2 = pd.concat([data_slide_features, tmp3], axis=1)
print(data_event2.shape)
print(data_event2.head())
