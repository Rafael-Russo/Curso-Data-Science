import Orange

base_credit = Orange.data.Table('./content/credit_data_regras.csv')

majority = Orange.classification.MajorityLearner()

previsoes = Orange.evaluation.testing.TestOnTestData(base_credit, base_credit, [majority])
print(Orange.evaluation.CA(previsoes))

# for registro in base_credit:
#     print(registro.get_class())

# from collections import Counter
# print(Counter(str(registro.get_class()) for registro in base_credit))