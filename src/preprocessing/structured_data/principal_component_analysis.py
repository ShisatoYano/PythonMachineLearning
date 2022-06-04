import tkinter as tk
import tkinter.filedialog as tkfd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

root = tk.Tk()
root.withdraw()

# select csv file
csv_path = tkfd.askopenfilename(filetypes=[("CSV", "*.csv")])
df = pd.read_csv(csv_path, sep=',')
# extract 'y'
y = df['y']

# normalize data
# z-score normalization
sc = StandardScaler()
sc.fit(df)
df_sc = pd.DataFrame(sc.transform(df), columns=df.columns)

# create new variable by principal component analysis
pca = PCA(0.80)
df_pca = pca.fit_transform(df_sc)
print(pca.n_components_)
print(df_pca)
df_pca = pd.DataFrame(df_pca)
df_pca['y'] = y
print(df_pca)

# draw scatter of 1st component and 2nd component about 'y'
df_pca_0 = df_pca[df_pca['y'] == 0]
df_pca_0 = df_pca_0.drop('y', axis=1)
plt.scatter(df_pca_0[0], df_pca_0[1], c="red", label=0)
df_pca_1 = df_pca[df_pca['y'] == 1]
df_pca_1 = df_pca_1.drop('y', axis=1)
plt.scatter(df_pca_1[0], df_pca_1[1], c="blue", label=1)
plt.legend()
plt.xlabel("1st-comp")
plt.ylabel("2nd-comp")
plt.show()
