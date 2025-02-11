import pickle
with open('./content/credit.pkl', mode='rb') as f:
    x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f)

from sklearn.tree import DecisionTreeClassifier
arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_credit.fit(x_credit_treinamento, y_credit_treinamento)
# print(arvore_credit.feature_importances_)

previsoes = arvore_credit.predict(x_credit_teste)

from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(y_credit_teste, previsoes))
print(classification_report(y_credit_teste, previsoes))

# Visualização gráfica da matriz de confusão (comparando se as previsões estão corretas)
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(arvore_credit)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)
cm.show()

# Visualização da árvore de decisões criada pelo algoritmo
from sklearn import tree
import matplotlib.pyplot as plt
previsores = ['income', 'age', 'loan']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
tree.plot_tree(arvore_credit, feature_names=previsores, class_names=[str(c) for c in arvore_credit.classes_], filled=True)
# Salva imagem da árvore
# figura.savefig('./content/credit_tree.png')
plt.show()