import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_census = pd.read_csv('./content/census.csv')

# Descrição da base de dados
# print(base_census.describe())

# Verificação de valores nullos
# print(base_census.isnull().sum())

# print(np.unique(base_census['income'], return_counts=True))

# sns.countplot(x = base_census['income'])

# plt.hist(x = base_census['age'])

# plt.hist(x = base_census['education-num'])

# plt.hist(x = base_census['hour-per-week'])

# plt.show()

# grafico = px.treemap(base_census, path=['workclass', 'age'])
# grafico.show()

# grafico = px.treemap(base_census, path=['occupation', 'relationship', 'age'])
# grafico.show()

# grafico = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])
# grafico.show()

# grafico = px.parallel_categories(base_census, dimensions=['workclass', 'occupation', 'income'])
# grafico.show()

# grafico = px.parallel_categories(base_census, dimensions=['education', 'income'])
# grafico.show()

x_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values

# Transformação dos atributos categóricos
# LabelEncoder - transforma valores string em numéricos para fazermos calculos com os algoritmos
from sklearn.preprocessing import LabelEncoder
# label_encoder_teste = LabelEncoder()
# print(x_census[:, 1])
# teste = label_encoder_teste.fit_transform(x_census[:, 1])
# print(teste)

labelEncoderWorkclass = LabelEncoder()
labelEncoderEducation = LabelEncoder()
labelEncoderMarital = LabelEncoder()
labelEncoderOccupation = LabelEncoder()
labelEncoderRelationship = LabelEncoder()
labelEncoderRace = LabelEncoder()
labelEncoderSex = LabelEncoder()
labelEncoderCountry = LabelEncoder()

# print(x_census[0])

x_census[:, 1] = labelEncoderWorkclass.fit_transform(x_census[:, 1])
x_census[:, 3] = labelEncoderEducation.fit_transform(x_census[:, 3])
x_census[:, 5] = labelEncoderMarital.fit_transform(x_census[:, 5])
x_census[:, 6] = labelEncoderOccupation.fit_transform(x_census[:, 6])
x_census[:, 7] = labelEncoderRelationship.fit_transform(x_census[:, 7])
x_census[:, 8] = labelEncoderRace.fit_transform(x_census[:, 8])
x_census[:, 9] = labelEncoderSex.fit_transform(x_census[:, 9])
x_census[:, 13] = labelEncoderCountry.fit_transform(x_census[:, 13])

# print(x_census[0])

# OneHotEncoder - técnica para os valores categóricos gerados pelo labelEncoder não viesem o algoritimo (cada tipo de dado irá virar uma coluna do tipo boolean)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
oneHotEncoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')

# print(x_census.shape)
x_census = oneHotEncoder_census.fit_transform(x_census).toarray()
# print(x_census.shape)

from sklearn.preprocessing import StandardScaler
scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)

from sklearn.model_selection import train_test_split
x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(x_census, y_census, test_size=0.15, random_state=0)

# print(x_census_treinamento.shape, y_census_treinamento.shape)
# print(x_census_teste.shape, y_census_teste.shape)

import pickle
with open('./content/census.pkl', mode='wb') as f:
    pickle.dump([x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste], f)

