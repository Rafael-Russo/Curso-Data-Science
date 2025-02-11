import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_risco_credito = pd.read_csv('./content/risco_credito.csv')

x_risco_credito = base_risco_credito.iloc[:, 0:4].values
y_risco_credito = base_risco_credito.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

labelEncoderHistoria = LabelEncoder()
labelEncoderDivida = LabelEncoder()
labelEncoderGarantias = LabelEncoder()
labelEncoderRenda = LabelEncoder()

x_risco_credito[:, 0] = labelEncoderHistoria.fit_transform(x_risco_credito[:, 0])
x_risco_credito[:, 1] = labelEncoderDivida.fit_transform(x_risco_credito[:, 1])
x_risco_credito[:, 2] = labelEncoderGarantias.fit_transform(x_risco_credito[:, 2])
x_risco_credito[:, 3] = labelEncoderRenda.fit_transform(x_risco_credito[:, 3])

import pickle
with open('./content/risco_credito.pkl', mode='wb') as f:
    pickle.dump([x_risco_credito, y_risco_credito], f)