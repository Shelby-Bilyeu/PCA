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

pca = PCA()
df.loc['3'] = 0
df_pca = pca.fit_transform(df)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_pca)
pcaData = pd.DataFrame(scaled_data)
pcaData.plot(subplots=True,figsize=(12,6))
pcaData.plot(figsize=(12,2))

