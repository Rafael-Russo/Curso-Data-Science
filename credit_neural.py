# Ler o arquivo pkl salvo
import pickle
with open('./content/credit.pkl', mode='rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

from sklearn.neural_network import MLPClassifier
rede_neural_credit = MLPClassifier(max_iter=1000, verbose=True, tol=0.000010, solver='adam', activation='relu', hidden_layer_sizes=(2,2))
rede_neural_credit.fit(x_credit_treinamento, y_credit_treinamento)

previsoes = rede_neural_credit.predict(x_credit_teste)
# print(previsoes)

# Porcentagem de acerto do algoritmo, matriz de confusão e métricas do desempenho do algoritmo
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_credit_teste, previsoes))
print(confusion_matrix(y_credit_teste, previsoes))
print(classification_report(y_credit_teste, previsoes))

# Visualização gráfica da matriz de confusão (comparando se as previsões estão corretas)
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(rede_neural_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
cm.show()