import Orange

base_risco_credito = Orange.data.Table('./content/risco_credito_regras.csv')

# print(base_risco_credito)
# print(base_risco_credito.domain.class_var.values)

cn2 = Orange.classification.rules.CN2Learner()
regras_risco_credito = cn2(base_risco_credito)

# for regras in regras_risco_credito.rule_list:
#     print(regras)

# Previsão dos registros:
#   [História Boa (0), Dívida Alta (0), Garantia Nenhuma (1) e Renda >35 (2)]
#   [História Ruim (2), Dívida Alta (0), Garantia Adequada (0) e Renda <15 (0)]
previsoes = regras_risco_credito([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])
for i in previsoes:
    print(base_risco_credito.domain.class_var.values[i])