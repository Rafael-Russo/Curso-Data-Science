import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_credit = pd.read_csv('./content/credit_data.csv')

# print(np.unique(base_credit['default'], return_counts=True))

# sns.countplot(x = base_credit['default'])

# plt.hist(x = base_credit['age'])

# plt.hist(x = base_credit['income'])

# plt.hist(x = base_credit['loan'])
# plt.show()

# grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
# grafico.show()

# print(base_credit.loc[base_credit['age'] < 0])
# print(base_credit[base_credit['age'] < 0])

# Apagando a coluna age inteira com valores inconsistentes
# base_credit2 = base_credit.drop('age', axis=1)

# Apagando as linhas com os valores inconsistentes
# base_credit3 = base_credit.drop(base_credit[base_credit['age'] < 0].index)

# Preencher valores inconsistentes com a média
base_credit['age'][base_credit['age'] > 0].mean()
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92

# Achando valores que são nullos
# print(base_credit.isnull().sum())
# print(base_credit.loc[pd.isnull(base_credit['age'])])

# Preenchendo esses valores com a média
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)
# print(base_credit.loc[pd.isnull(base_credit['age'])])

# Busca de várias linhas por ID
# print(base_credit.loc[base_credit['clientid'].isin([29, 31, 32])])

# Váriaveis previsoras - X // Váriveis de classe - Y
x_credit = base_credit.iloc[:, 1:4].values
y_credit = base_credit.iloc[:, 4].values

# Navegando dentro das variáveis
# print(x_credit[:, 0].min(), x_credit[:, 1].min(), x_credit[:, 2].min())
# print(x_credit[:, 0].max(), x_credit[:, 1].max(), x_credit[:, 2].max())

# Padronização (Standardisation) X = (X - Média(X)) / (Desvio padrão(X))
# Normalização (Normalization) X = (X - Min(X)) / (Max(X) - Min(X))

# Padronização de X para não viesar a decisão da IA dependendo do modelo
from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()
x_credit= scaler_credit.fit_transform(x_credit)

# Separação dos dados em base de treinamento e base de teste nos dois eixos X e Y
from sklearn.model_selection import train_test_split
x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(x_credit, y_credit, test_size=0.25, random_state=0)

# print(x_credit_treinamento.shape, y_credit_treinamento.shape)
# print(x_credit_teste.shape, y_credit_teste.shape)

# Salvar base de dados de teste e de treinamento
import pickle
with open('./content/credit.pkl', mode='wb') as f:
    pickle.dump([x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f)
