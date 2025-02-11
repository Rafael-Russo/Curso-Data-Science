import pickle
with open('./content/risco_credito.pkl', mode='rb') as f:
    x_risco_credito, y_risco_credito = pickle.load(f)

from sklearn.tree import DecisionTreeClassifier
arvore_risco_credito = DecisionTreeClassifier(criterion='entropy')
arvore_risco_credito.fit(x_risco_credito, y_risco_credito)
# print(arvore_risco_credito.feature_importances_)

# Visualização da árvore de decisões criada pelo algoritmo
# from sklearn import tree
# import matplotlib.pyplot as plt
# previsores = ['história', 'dívida', 'garantia', 'renda']
# figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
# tree.plot_tree(arvore_risco_credito, feature_names=previsores, class_names=arvore_risco_credito.classes_, filled=True)
# plt.show()

# Previsão dos registros:
#   [História Boa (0), Dívida Alta (0), Garantia Nenhuma (1) e Renda >35 (2)]
#   [História Ruim (2), Dívida Alta (0), Garantia Adequada (0) e Renda <15 (0)]
previsoes = arvore_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(previsoes)