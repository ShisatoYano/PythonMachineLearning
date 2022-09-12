import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from keras.models import Sequential
from keras.layers import Dense

data_df = pd.read_csv("/workspaces/PythonMachineLearning/preprocess_sample_data/chap6/data/data/energydata.csv", sep=',')[["date", "Appliances"]]
data_df["date"] = pd.to_datetime(data_df["date"], format="%Y-%m-%d %H:%M:%S")
print(data_df.head())

# plt.plot(data_df["date"], data_df["Appliances"])
# plt.xlabel("date")
# plt.xticks(rotation=30)
# plt.ylabel("data1")
# plt.show()

train = data_df[data_df["date"] < "2016-04-11 17:00:00"]
test = data_df[data_df["date"] >= "2016-04-11 17:00:00"]
print(train.shape, test.shape)

mc = MinMaxScaler()
train = mc.fit_transform(train[["Appliances"]])
test = mc.fit_transform(test[["Appliances"]])
print(train)
print(test)

width = 144
train, test = train.flatten(), test.flatten()
train_vec, test_vec = [], []
for i in range(len(train) - width):
    train_vec.append(train[i:i + width])
print(pd.DataFrame(train_vec).shape)
print(pd.DataFrame(train_vec).head())
for i in range(len(test) - width):
    test_vec.append(test[i:i + width])
print(pd.DataFrame(test_vec).shape)
print(pd.DataFrame(test_vec).head())

# k-nearest neighbor
train_vec = np.array(train_vec)
test_vec = np.array(test_vec)
model = NearestNeighbors(n_neighbors=1)
model.fit(train_vec)

dist, _ = model.kneighbors(test_vec)
dist = dist / np.max(dist)
# plt.plot(dist)
# plt.show()

# autoencoder
model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(144,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(144, activation="sigmoid"))
model.summary()

# training
model.compile(loss="mse", optimizer="adam")
hist = model.fit(train_vec, train_vec, batch_size=128,
                 verbose=1, epochs=20, validation_split=0.2)

# show error per epoch
# plt.plot(hist.history["loss"], label="loss")
# plt.plot(hist.history["val_loss"], label="val_loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.legend()

# predict
pred = model.predict(test_vec)
plt.plot(test_vec[:, 0], label="test")
plt.plot(pred[:, 0], label="pred")
plt.legend()

plt.show()
