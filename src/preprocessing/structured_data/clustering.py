import tkinter as tk
import tkinter.filedialog as tkfd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()

# select csv file
csv_path = tkfd.askopenfilename(filetypes=[("CSV", "*.csv")])
bank_df = pd.read_csv(csv_path, sep=',')

# normalize data
# z-score normalization
sc = StandardScaler()
sc.fit(bank_df)
bank_df_sc = pd.DataFrame(sc.transform(bank_df), columns=bank_df.columns)

# hierarchical clustering by ward's method
hcls = linkage(bank_df_sc, metric="euclidean", method="ward")
# dendrogram(hcls)
# cluster id
cst_group = fcluster(hcls, 100, criterion="distance")
print(cst_group)

# non-hierarchical clustering by k-means method
kcls = KMeans(n_clusters=10)
cst_group_k = kcls.fit_predict(bank_df_sc)
print(cst_group_k)
for i in range(10):
    labels = bank_df_sc[cst_group_k == i]
    plt.scatter(labels["age"], labels["balance"], label=i)
plt.legend()
plt.xlabel("age")
plt.ylabel("balance")

plt.show()
