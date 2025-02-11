# Ler o arquivo pkl salvo
import pickle


with open('./content/credit.pkl', mode='rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

from sklearn.ensemble import RandomForestClassifier
random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
random_forest_credit.fit(x_credit_treinamento, y_credit_treinamento)
previsoes = random_forest_credit.predict(x_credit_teste)

# Porcentagem de acerto do algoritmo, matriz de confusão e métricas do desempenho do algoritmo
from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_credit_teste, previsoes))
print(classification_report(y_credit_teste, previsoes))

# Visualização gráfica da matriz de confusão (comparando se as previsões estão corretas)
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(random_forest_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
cm.show()