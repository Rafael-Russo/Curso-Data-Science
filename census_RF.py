import pickle
with open('./content/census.pkl', mode='rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

from sklearn.ensemble import RandomForestClassifier
arvore_census = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
arvore_census.fit(x_census_treinamento, y_census_treinamento)
# print(arvore_credit.feature_importances_)

previsoes = arvore_census.predict(x_census_teste)

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_census_teste, previsoes))
print(classification_report(y_census_teste, previsoes))

# Visualização gráfica da matriz de confusão (comparando se as previsões estão corretas)
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(arvore_census)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)
cm.show()