import pickle
with open('./content/census.pkl', mode='rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

from sklearn.neural_network import MLPClassifier
rede_neural_census = MLPClassifier(max_iter=1000, verbose=True, tol=0.000010, solver='adam', activation='relu', hidden_layer_sizes=(55,55))
rede_neural_census.fit(x_census_treinamento, y_census_treinamento)
previsoes = rede_neural_census.predict(x_census_teste)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_census_teste, previsoes))
print(confusion_matrix(y_census_teste, previsoes))
print(classification_report(y_census_teste, previsoes))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(rede_neural_census)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)
cm.show()