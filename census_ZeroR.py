import Orange

base_census = Orange.data.Table('./content/census_regras.csv')

majority = Orange.classification.MajorityLearner()

previsoes = Orange.evaluation.testing.TestOnTestData(base_census, base_census, [majority])
print(Orange.evaluation.CA(previsoes))

# for registro in base_census:
#     print(registro.get_class())

# from collections import Counter
# print(Counter(str(registro.get_class()) for registro in base_census))