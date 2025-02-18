import numpy as np

import pickle
with open('./content/risco_credito.pkl', mode='rb') as f:
    x_risco_credito, y_risco_credito = pickle.load(f)

x_risco_credito = np.delete(x_risco_credito, [2, 7, 11], axis=0)
y_risco_credito = np.delete(y_risco_credito, [2, 7, 11], axis=0)

from sklearn.linear_model import LogisticRegression
logistic_risco_credito = LogisticRegression(random_state=0)
logistic_risco_credito.fit(x_risco_credito, y_risco_credito)

print(logistic_risco_credito.intercept_)
print(logistic_risco_credito.coef_)

# Previsão dos registros:
#   [História Boa (0), Dívida Alta (0), Garantia Nenhuma (1) e Renda >35 (2)]
#   [História Ruim (2), Dívida Alta (0), Garantia Adequada (0) e Renda <15 (0)]
previsoes = logistic_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(previsoes)