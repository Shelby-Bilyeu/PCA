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
