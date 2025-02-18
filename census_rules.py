#
# BASE DE DADOS DO CENSUS É MUITO GRANDE, ESSE ALGORITMO DEMORA MUITO TEMPO, NA AULA ELE USA O APLICATIVO DESKTOP DO ORANGE
#

import Orange

base_census = Orange.data.Table('./content/census_regras.csv')

base_dividida = Orange.evaluation.testing.sample(base_census, n=0.15)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

cn2 = Orange.classification.rules.CN2Learner()
regras_census = cn2(base_treinamento)

# for regras in regras_census.rule_list:
#     print(regras)

previsoes = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: base_census])
print(Orange.evaluation.CA(previsoes))