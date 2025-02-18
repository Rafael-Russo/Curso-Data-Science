import pickle
with open('./content/census.pkl', mode='rb') as f:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f)

from sklearn.svm import SVC
svm_census = SVC(kernel='linear', random_state=0, C=1.0)
svm_census.fit(x_census_treinamento, y_census_treinamento)
previsoes = svm_census.predict(x_census_teste)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_census_teste, previsoes))
print(confusion_matrix(y_census_teste, previsoes))
print(classification_report(y_census_teste, previsoes))

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(svm_census)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)
cm.show()