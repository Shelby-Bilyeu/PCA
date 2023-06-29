import numpy as np
import pandas as pd
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


#excel import requires fancy openpyx1
!pip install --upgrade openpyx1

# import ecg data
!wget -O data_ecg.xlsx https://github.com/Shelby-Bilyeu/PCA/blob/main/ecg_data.xlsx




headers = ["1","2","3"]
df = pd.read_excel('data_ecg.xlsx', names=headers, header=None)
df.plot(subplots=True,figsize=(12,6))
df.plot(figsize=(12,2))

df.loc['column 3'] = 0
scaler = StandardScaler()
scaler.fit(df)
df_scaled = scaler.transform(df)
pca = PCA(n_components=3)
pca.fit(df_scaled)
df_pca = pca.transform(df_scaled)
pcaData = pd.DataFrame(df_pca)
pcaData.columns = ["component1", "component2", "component3"]
pcaData.plot(subplots=True,figsize=(12,6))
pcaData.plot(figsize=(12,2))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Xax = pcaData["component1"]
Yax = pcaData["component2"]
Zax = pcaData["component3"]

ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_zlabel("PC3", fontsize=12)

ax.scatter(Xax, Yax, Zax)

origin = [0, 0, 0]  # Origin of the axes
axes_lengths = [max(Xax) - min(Xax), max(Yax) - min(Yax), max(Zax) - min(Zax)]  # Lengths of the axes
ax.plot([origin[0], origin[0] + axes_lengths[0]], [origin[1], origin[1]], [origin[2], origin[2]], 'r')
ax.plot([origin[0], origin[0]], [origin[1], origin[1] + axes_lengths[1]], [origin[2], origin[2]], 'r')
ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2] + axes_lengths[2]], 'r')

plt.title("3D PCA plot")
plt.show()

from sklearn.decomposition import FastICA
df.loc['column 3'] = 0
scaler = StandardScaler()
scaler.fit(df)
df_scaled = scaler.transform(df)

ica = FastICA(n_components=3)
icaData = ica.fit_transform(df_scaled)

icaData = pd.DataFrame(icaData, columns=["component1", "component2", "component3"])

icaData.plot(subplots=True, figsize=(12, 6))
icaData.plot(figsize=(12, 2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Xax = icaData["component1"]
Yax = icaData["component2"]
Zax = icaData["component3"]

ax.set_xlabel("PC1", fontsize=12)
ax.set_ylabel("PC2", fontsize=12)
ax.set_zlabel("PC3", fontsize=12)

ax.scatter(Xax, Yax, Zax)

origin = [0, 0, 0]  # Origin of the axes
axes_lengths = [max(Xax) - min(Xax), max(Yax) - min(Yax), max(Zax) - min(Zax)]  # Lengths of the axes
ax.plot([origin[0], origin[0] + axes_lengths[0]], [origin[1], origin[1]], [origin[2], origin[2]], 'r')
ax.plot([origin[0], origin[0]], [origin[1], origin[1] + axes_lengths[1]], [origin[2], origin[2]], 'r')
ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2] + axes_lengths[2]], 'r')

plt.title("3D ICA plot")
plt.show()

