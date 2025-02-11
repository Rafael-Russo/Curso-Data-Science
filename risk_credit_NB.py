import pickle
with open('./content/risco_credito.pkl', mode='rb') as f:
    x_risco_credito, y_risco_credito = pickle.load(f)

# Utilização do algoritimo Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive_risco_credito = GaussianNB()
naive_risco_credito.fit(x_risco_credito, y_risco_credito)

# Previsão dos registros:
#   [História Boa (0), Dívida Alta (0), Garantia Nenhuma (1) e Renda >35 (2)]
#   [História Ruim (2), Dívida Alta (0), Garantia Adequada (0) e Renda <15 (0)]
previsao = naive_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(previsao)

print(naive_risco_credito.classes_)
print(naive_risco_credito.class_count_)
print(naive_risco_credito.class_prior_)