# Ler o arquivo pkl salvo
import pickle
with open('./content/credit.pkl', mode='rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

from sklearn.linear_model import LogisticRegression
logistic_credit = LogisticRegression(random_state=0)
logistic_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes = logistic_credit.predict(x_credit_teste)
# print(previsoes)

# Porcentagem de acerto do algoritmo, matriz de confusão e métricas do desempenho do algoritmo
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_credit_teste, previsoes))
print(confusion_matrix(y_credit_teste, previsoes))
print(classification_report(y_credit_teste, previsoes))

# Visualização gráfica da matriz de confusão (comparando se as previsões estão corretas)
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(logistic_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
cm.show()