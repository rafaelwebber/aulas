import numpy as np
import skfuzzy as ctrl

peso = ctrl.Antecedent(np.arange(20,120,1), 'peso')
altura = ctrl.Antecedent(np.arange(1.2,2,0.1), 'altura')

tamanho = ctrl.Antecedent(np.arange(0,1,0.1), 'tamanho')