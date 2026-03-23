import numpy as np  
import pandas as pd 
import random as rd 
from random import randint 
import matplotlib.pyplot as plt

n = 10 
numero_itens = np.arange(1, n+1)

pesos = [2.5, 1.8, 0.7, 2.1, 1.5, 2.2, 0.9, 1.6, 0.5, 1.1]

valore = [2000, 1450, 3400, 1900, 1300, 1000, 600, 1300, 400, 900]

nomes = [ 
    "Notebook", 
    "Smartphone",
    "Tablet", 
    "Monitor", 
    "Teclado",
    "Mouse", 
    "Fone de Ouvido", 
    "Impressora", 
    "Webcam", 
    "Roteador"
    "Caixa de Som"
]

max_peso_mochila = 7

for i in range (numero_itens.shape[0]):
    print ('==================================')
    print ('Item: {}, \nPeso: {}, \nValor: {}'.format(nomes[i], pesos[i], valore[i]))
    print ('==================================')

solucao_por_populacao = 8
tamanho_populacao = (solucao_por_populacao, numero_itens.shape[0])

print ('==================================')
print ("\n")
print('Tamanho da população: {}'.format(tamanho_populacao))
print('Numero de individuos (solução) = {}'.format(solucao_por_populacao))
print('Numero de itens (genes) = {}'.format(tamanho_populacao[1]))
print ("\n")
print ('==================================')
print ("\n")
n_geracoes = 10

populacao_inicial = np.eye(tamanho_populacao[0], tamanho_populacao[1], k=0)

populacao_inicial = populacao_inicial.astype(int)

print ('==================================')
print ("\n")
print('População Inicial: \n{}'.format(populacao_inicial))
print ("\n")
print ('==================================')

def cal_fitness(peso, valor, populacao, max_peso_mochila):
    fitness = np.zeros(populacao.shape[0])

    for i in range(populacao.shape[0]):

        S1 = np.sum(populacao[i] * valor)
        S2 = np.sum(populacao[i] * peso)

        if S2 <= max_peso_mochila:
            fitness[i] = S1

        else:
            fitness[i] = 0

    return fitness.astype(float)

def selecao_roleta(fitness, numero_pais, populacao):
    max_fitness = sum(fitness)

    probabilidades = fitness / max_fitness

    selecionados = populacao[np.random.choice(len(populacao), size=numero_pais, p=probabilidades)]
    return selecionados

def crossover(pais, numero_filhos):
    filhos = np.zeros((numero_filhos, pais.shape[1]))
    ponto_crossover = int(pais.shape[1] / 2)

    for k in range(numero_filhos):
        pai_1_idx = k%pais.shape[0]

        pai_2_idx = (k+1)%pais.shape[0]

        filhos[k, 0: ponto_crossover] = pais[pai_1_idx, 0:ponto_crossover]
        filhos[k, ponto_crossover:] = pais[pai_2_idx, ponto_crossover:]
    return filhos

def mutacao(filhos):
    mutacoes = filhos 

    for i in range(mutacoes.shape[0]):
        posicao_gene = randint(0, filhos.shape[1]-1)

        if mutacoes[i, posicao_gene] == 0:
            mutacoes[i,posicao_gene] = 1

        else: 
            mutacoes[i,posicao_gene] = 0   

    return mutacoes 

def rodar_AG(pesos, valores, populacao, tamanho_populacao, n_geracoes, max_peso_mochila):
    historico_fitness, historico_popucao = [], []

    numero_pais = int(tamanho_populacao[0] / 2)

    numero_filhos = tamanho_populacao[0]-numero_pais
    fitness = []

    for i in range(n_geracoes):
        print ('==================================')
        print('*---Começando a geração {}---*'.format(i))
        print ('==================================')
        fitness = cal_fitness(pesos, valores, populacao, max_peso_mochila)

        historico_fitness.append(fitness.copy())
        historico_popucao.append(populacao.copy())

        pais = selecao_roleta(fitness, numero_pais, populacao)

        filhos = crossover(pais, numero_filhos) 
        filhos_mutados = mutacao(filhos)
        print('População Antiga: ')
        print(populacao)
        populacao[0:pais.shape[0], :] = pais
        populacao[pais.shape[0]:, :] = filhos_mutados
        print("\n")
        print('População Nova: ')
        print(populacao)
        print("\n")

    return historico_fitness, historico_popucao

historico_fitness, historico_popucao = rodar_AG(pesos, valore, populacao_inicial, tamanho_populacao, n_geracoes, max_peso_mochila) 

dataframe = pd.DataFrame(historico_fitness)

dataframe

max_index = dataframe.values.argmax()
linha, coluna = np.unravel_index(max_index, dataframe.shape)

print("Valor do fitness máximo: ", dataframe.iloc[linha, coluna])
print("Linha do maior Fitness (geração): ", linha)
print("Coluna do maior Fitness (indivíduo): ", coluna)

melhor_individuo = historico_popucao[linha][coluna]

itens_selecionados = numero_itens * melhor_individuo
dataframe_itens = pd.DataFrame(columns=['Item', 'Valor', 'Peso'])

for i in itens_selecionados:
    if i != 0:
        posicao = i -1
        item = {'item': nomes[posicao], 'Valor': valore[posicao], 'Peso': pesos[posicao]}
        dataframe_itens.loc[len(dataframe_itens)] = item

fitness_medio = [np.mean(fitness) for fitness in historico_fitness]
fitness_max = [np.max(fitness) for fitness in historico_fitness]

plt.plot(list(range(n_geracoes)), fitness_medio, label = 'Fitness Médio')
plt.plot(list(range(n_geracoes)), fitness_max, label = 'Fitness Máximo')
plt.legend()
plt.title('Fitness ao decorrer das gerações')
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.show()
print(np.asanyarray(historico_fitness).shape)