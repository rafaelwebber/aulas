import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import skfuzzy.membership as mf

# Variáveis de entrada
peso = ctrl.Antecedent(np.arange(20,120,1), 'peso')
altura = ctrl.Antecedent(np.arange(1.2,2,0.1), 'altura')

# Variável de saída
tamanho = ctrl.Consequent(np.arange(0,1,0.1), 'tamanho')

# peso
peso['baixo'] = mf.trimf(peso.universe, [20, 45, 70])
peso['medio'] = mf.trimf(peso.universe, [55, 75, 95])
peso['alto'] = mf.trimf(peso.universe, [80, 100, 120])

peso.view()

# altura
altura['baixo'] = mf.trapmf(altura.universe, [1.2, 1.2, 1.4, 1.5])
altura['medio'] = mf.trapmf(altura.universe, [1.4, 1.5, 1.6, 1.7])
altura['alto'] = mf.trapmf(altura.universe, [1.68, 1.7, 2, 2])

altura.view()

# tamanho
tamanho['pequeno'] = mf.trapmf(tamanho.universe, [0, 0, 0.4, 0.5])
tamanho['medio'] = mf.trapmf(tamanho.universe, [0.4, 0.5, 0.6, 0.7])
tamanho['grande'] = mf.trapmf(tamanho.universe, [0.65, 0.7, 1, 1])

tamanho.view()

# regras
regra1 = ctrl.Rule(peso['baixo'], tamanho['pequeno'])

regra2 = ctrl.Rule(altura['medio'] | peso['medio'], tamanho['medio'])

regra3 = ctrl.Rule(altura['alto'] & peso['alto'], tamanho['grande'])

# controlador fuzzy
fuzzy_ctrl = ctrl.ControlSystem([regra1, regra2, regra3])

# motor de inferência
engine = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# entradas
engine.input['peso'] = 60
engine.input['altura'] = 1.70

# cálculo
engine.compute()

print("Tamanho calculado:", engine.output['tamanho'])

# gráfico do resultado
tamanho.view(sim=engine)

input("")