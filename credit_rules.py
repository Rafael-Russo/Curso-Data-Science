import Orange

base_credit = Orange.data.Table('./content/credit_data_regras.csv')

base_dividida = Orange.evaluation.testing.sample(base_credit, n=0.25)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

cn2 = Orange.classification.rules.CN2Learner()
regras_credit = cn2(base_treinamento)

# for regras in regras_credit.rule_list:
#     print(regras)

previsoes = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: regras_credit])
print(Orange.evaluation.CA(previsoes))