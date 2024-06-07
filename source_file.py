############################################################################################################
# Este ficheiro é fruto dos objetivos estabelecidos do Guião para preparação do 2º trabalho de EDA 23/24
# (aula de apoio de 9/Maio/2024) para gerarem um conjuntos de dados (datasets).
############################################################################################################

import pandas as pd
import random

# pandas dataframe que vai conter os nomes do ficheiro disponibilizado para o trabalho
names_df = pd.read_csv('nomes.csv', sep=',')

# decidimos guardar o nº de transações numa variável
# (está em letras maiúsculas para dar o sentido de constante)
#
# Desta forma damos mais interpretação ao nº e damos a entender melhor
# o que pretendemos fazer
#  Nota: usámos "_" nos números grandes deste ficheiro.
# É uma notação disponilizada pelo python para se perceber melhor as casas decimais envolvidas.
# referÊncia: https://peps.python.org/pep-0515/
NUMBER_OF_TRANSACTIONS = 10_000

# listas que irão conter as respectivas informações das transações
senders_list = list()
receivers_list = list()
values_list = list()


# loop para criar as 10000 transações escolhendo aleatoriamente 2 nomes
# que não sejam iguais entre si:
#   Nota: usamos "_" pois não queremos usar o iterador para atingir o objetivo
for _ in range(NUMBER_OF_TRANSACTIONS):

    # Aqui o método sample() do pandas é usado para selecionar um elemento (linha) aleatória
    # da coluna 'Nomes' do DataFrame names_df.
    # O atributo "values" converte a linha resultante num array NumPy.
    # e o "[0]" selecciona o 1º elemento desse array, que é onde está o nome.
    sender_name = names_df['Nomes'].sample().values[0]

    # Este ciclo serve para garantir que o nome do receiver não o mesmo que o sender
    # Ele escolhe um nome aleatório até este ser diferente do nome do sender e sai
    # do ciclo quando essa condição for verdadeira
    while True:
        receiver_name = names_df['Nomes'].sample().values[0]
        if receiver_name != sender_name:
            break

    # guardar os nomes selecionados nas respectivas listas
    senders_list.append(sender_name)
    receivers_list.append(receiver_name)

    # guardar um valor aleatoriamente gerado, usando a biblioteca random, entre 1 e 100000
    values_list.append(random.randint(1, 100_000))


# dicionário que irá conter a informação para criar um pandas.dataframe
dict_of_transactions = {
    'sender': senders_list,
    'receiver': receivers_list,
    'value': values_list
}


# gerar o ficheiro csv que vai conter as 10000 transações
pd.DataFrame(dict_of_transactions).to_csv('transactions.csv', index=False)

# mensagem para se perceber que o programa terminou e dar a entender os resultados foram alcançados
print("O ficheiro CSV foi gerado com sucesso! Pode encontrar o mesmo na pasta/diretório onde chamou este script.")
