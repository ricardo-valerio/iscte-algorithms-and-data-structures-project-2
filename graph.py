import csv
import pprint


class Node:

    def __init__(self, key, payload=None):
        self._key = key
        self._payload = payload

    def get_key(self):
        return self._key

    def get_info(self):
        return self._payload

    def __hash__(self):
        return hash(self.get_key())


class Edge:
    def __init__(self, start, end, weight=None):
        self._start = start
        self._end = end
        self._weight = weight

    def get_nodes(self):
        return self._start, self._end

    def get_info(self):
        return self._weight

    def opposite(self, v):
        """
        método que vai devolver o nó oposto da aresta
        """
        if v == self._start:
            return self._end
        else:
            return self._start

    def __hash__(self):
        return hash(self.get_nodes())

    def __str__(self):
        """
            Retorna uma representação em string da aresta.
        """
        return f"Edge from {self._start} to {self._end} with weight {self._weight}"

    def __repr__(self):
        """
            Retorna uma representação em string da aresta para debug.
        """
        return self.__str__()


class BankTransfersGraph:

    # métodos pedidos na página 2 do enunciado: ----------------------------------------------------------------------
    def __init__(self):
        """
            Inicializa o grafo de transações bancárias.
            Cria um dicionário para armazenar os nós e inicializa contadores para nós, arestas e transações.
        """
        self._graph = {}
        self._node_counter = 0
        self._edge_counter = 0
        self._transaction_counter = 0

    def load_file(self, file_path):
        """
            Carrega as transações a partir de um ficheiro CSV e adiciona ao grafo.
        """
        with open(file_path, 'r') as file:
            transactions = csv.reader(file)
            next(transactions)  # descartar o header
            for row in transactions:
                sender, receiver, value = row
                self.add_edge(sender, receiver, amount=int(value))

    def add_node(self, id):
        """
            Adiciona um nó ao grafo se ele ainda não existir.
        """
        if id not in self._graph.keys():
            node = self.create_node(id)
            self._graph[node.get_key()] = {
                "connections": [],
                "sent_to": {},
                "received_from": {},
                "number_of_transactions_sent_to": {},
                "number_of_transactions_received_from": {},
            }

            self._node_counter = self._node_counter + 1

    def create_node(self, id):
        """
            Cria um novo nó com o ID fornecido.
        """
        return Node(id)

    def add_edge(self, sender_id, receiver_id, amount):
        """
            Adiciona uma aresta (transação) entre dois nós existentes no grafo.
        """
        if sender_id in self._graph.keys() and \
                receiver_id in self._graph.keys() and \
                receiver_id in self._graph[sender_id]["sent_to"]:  # verifica se sender_id já enviou para receiver_id

            # Os dois nomes existem no grafo, portanto temos de verificar se estão conectados
            # isto é, se são vizinhos

            # DEBUG
            # print("PASSEI AQUI e uma transação com esses nomes já existe no grafo:", sender_id, receiver_id)
            # print("TOCA A SOMAR OS VALORES")

            # Somar os valores guardados em ambas as pessoas

            self._graph[sender_id]["sent_to"][receiver_id] = self._graph[sender_id]["sent_to"][receiver_id] + amount

            # Guardar informação discreta sobre o nº de transações envolvida entre os intervenientes
            if receiver_id not in self._graph[sender_id]["number_of_transactions_sent_to"]:
                self._graph[sender_id]["number_of_transactions_sent_to"][receiver_id] = 1
            else:
                self._graph[sender_id]["number_of_transactions_sent_to"][receiver_id] += 1

            self._graph[receiver_id]["received_from"][sender_id] = self._graph[receiver_id]["received_from"][
                                                                       sender_id] + amount

            # Guardar informação discreta sobre o nº de transações envolvida entre os intervenientes
            if sender_id not in self._graph[receiver_id]["number_of_transactions_received_from"]:
                self._graph[receiver_id]["number_of_transactions_received_from"][sender_id] = 1
            else:
                self._graph[receiver_id]["number_of_transactions_received_from"][sender_id] += 1

            connection = Edge(sender_id, receiver_id, amount)
            self._graph[sender_id]["connections"].append(connection)
            self._graph[receiver_id]["connections"].append(connection)

            self._transaction_counter = self._transaction_counter + 1

        else:
            if sender_id not in self._graph.keys():  # Caso não exista node1 é criado esse nó
                self.add_node(sender_id)

            if receiver_id not in self._graph.keys():  # Caso não exista node2 é criado esse nó
                self.add_node(receiver_id)

            connection = Edge(sender_id, receiver_id, amount)  # Criar objeto do tipo Edge

            self._graph[sender_id]["connections"].append(connection)
            self._graph[sender_id]["sent_to"][receiver_id] = amount

            self._graph[receiver_id]["connections"].append(connection)
            self._graph[receiver_id]["received_from"][sender_id] = amount

            # Guardar informação discreta sobre o nº de transações envolvida entre os intervenientes
            if receiver_id not in self._graph[sender_id]["number_of_transactions_sent_to"]:
                self._graph[sender_id]["number_of_transactions_sent_to"][receiver_id] = 1
            else:
                self._graph[sender_id]["number_of_transactions_sent_to"][receiver_id] += 1

            if sender_id not in self._graph[receiver_id]["number_of_transactions_received_from"]:
                self._graph[receiver_id]["number_of_transactions_received_from"][sender_id] = 1
            else:
                self._graph[receiver_id]["number_of_transactions_received_from"][sender_id] += 1

            self._edge_counter = self._edge_counter + 1
            self._transaction_counter = self._transaction_counter + 1

        # DEBUG para ver o dicionário a crescer com as transações
        # pprint.pprint(self._graph)
        # print("=========================================================================")

    def get_edges(self, node, out=True):
        if node in self._graph.keys():
            connections = self._graph[node]['connections']
            if out:
                return [edge for edge in connections if edge.get_nodes()[0] == node]
            else:
                return [edge for edge in connections if edge.get_nodes()[1] == node]
        else:
            raise ValueError('The node is not on this graph')

    def nr_nodes(self):
        """
            Obtém o número total de nós presentes no grafo
        """
        return self._node_counter

    def nr_edges(self):
        """
            Obtém o número total de arestas presentes no grafo
        """
        return self._edge_counter

    def neighbours(self, node, out=True):
        """
            Obtém os nós vizinhos de um dado nó, se este existir no grafo, tanto de entrada ou de saíde, conforme o valor do parâmetro "out"
            Por default retornará os vizinhos de saída. Ou seja, as pessoas a quem enviou dinheiro.
        """

        if node not in self._graph:
            raise ValueError('The node is not in this graph')

        neighbours_list = []

        if out:
            neighbours_list.extend(self._graph[node]["sent_to"].keys())
        else:
            neighbours_list.extend(self._graph[node]["received_from"].keys())

        return list(set(neighbours_list))

    def avg_edges(self):
        """
            Obtém o número médio de arestas presentes no grafo
        """
        total_edges = sum(len(each_node_info["sent_to"]) for each_node_info in self._graph.values())
        total_nodes = len(self._graph.keys())
        return total_edges / total_nodes

    def plot_graph(self):
        """
            Método responsável por "desenhar" todo o grafo.
        """

        # Duas possíveis implementações:

        # 1 - biblioteca networkx ---------------------------------------------------------------------------
        import networkx as nx
        import matplotlib.pyplot as plt

        # Cria um dígrafo
        G = nx.DiGraph()

        # Adiciona arestas e nós ao grafo
        for node, data in self._graph.items():
            G.add_node(node)
            for receiver, amount in data['sent_to'].items():
                G.add_edge(node, receiver, weight=amount)
            for sender, amount in data['received_from'].items():
                G.add_edge(sender, node, weight=amount)

        pos = nx.spiral_layout(G)

        # Plot the graph
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), label_pos=0.33)
        nx.draw(G, pos, with_labels=True,
                node_size=800,
                node_color='skyblue',
                font_size=6,
                font_weight='bold',
                arrows=True,
                connectionstyle='arc3, rad = 0.1')
        plt.title("Grafo Orientado que representa todas as Transações")
        plt.show()

        # ---------------------------------------------------------------------------------------------------

        # 2 - biblioteca Jaal -------------------------------------------------------------------------------
        #      - https://mohitmayank.com/jaal/
        #      - https://github.com/imohitmayank/jaal
        #      - https://towardsdatascience.com/introducing-jaal-interacting-with-network-made-easy-124173bb4fa

        # from jaal import Jaal
        # import pandas as pd
        # transactions_dataframe = pd.read_csv("transactions.csv", sep=",")
        # transactions_dataframe.columns = ["from", "to", "weight"]
        # transactions_dataframe = transactions_dataframe.groupby(['from', 'to']).apply(set).apply(", ".join).reset_index()
        # Jaal(transactions_dataframe).plot(directed=True)
        # ----------------------------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------------

    # métodos que são sugeridos no guião de preparação para o trabalho: ------------------------------------------------------------------------------------------------------------------
    def most_connected_node(self):
        """
            Obtém o nó com mais ligações (de envio e de receção)
        """
        max_degree = 0
        most_connected_node = None

        for node in self._graph.keys():
            degree = self.in_degree(node) + self.out_degree(node)
            if degree > max_degree:
                max_degree = degree
                most_connected_node = node

        return most_connected_node

    def least_connected_node(self):
        """
            Obtém o nó com menos ligações (de envio e de receção)
        """
        min_degree = float('inf')  # Inicializa o menor grau com o maior valor possível
        least_connected_node = None

        for node in self._graph.keys():
            degree = self.in_degree(node) + self.out_degree(node)
            if degree < min_degree:
                min_degree = degree
                least_connected_node = node

        return least_connected_node

    def edge_with_highest_weight(self):
        """
            Obtém a aresta/transação com maior peso/valor
        """
        max_weight = float('-inf')  # Inicializa o peso máximo com o menor valor possível
        max_weight_edge = (None, None, 0)  # Armazena (nó de origem, nó de destino, peso)

        for node in self._graph.keys():
            for receiver, weight in self._graph[node]['sent_to'].items():
                if weight > max_weight:
                    max_weight = weight
                    max_weight_edge = (node, receiver, weight)

        return max_weight_edge

    def edge_with_lowest_weight(self):
        """
            Obtém a aresta/transação com menos peso/valor
        """
        min_weight = float('inf')  # Inicializa o peso mínimo com o maior valor possível
        min_weight_edge = (None, None, 0)  # Armazena (nó de origem, nó de destino, peso)

        for node in self._graph.keys():
            for receiver, weight in self._graph[node]['sent_to'].items():
                if weight < min_weight:
                    min_weight = weight
                    min_weight_edge = (node, receiver, weight)

        return min_weight_edge

    # ----------------------------------------------------------------------------------------------------------------

    # Outros métodos auxiliares que usamos para o métodos "most_connected_node" e "least_connected_node" -------------
    # e que podem vir a ajudar para fazer os métodos pedidos na página 3
    def in_degree(self, node):
        """
            Obtém o grau de entrada de um nó.
        """
        if node in self._graph.keys():
            return len(self._graph[node]["received_from"].keys())
        else:
            raise ValueError('The node is not on this graph')

    def out_degree(self, node):
        """
            Obtém o grau de saída de um nó.
        """
        if node in self._graph.keys():
            return len(self._graph[node]["sent_to"].keys())
        else:
            raise ValueError('The node is not on this graph')

    def nr_transactions(self):
        """
            Obtém o número total de transações no grafo.
        """
        return self._transaction_counter

    # ----------------------------------------------------------------------------------------------------------------

    # métodos que são pedidos na página 3 ----------------------------------------------------------------------------
    def visualize_node(self, node=None, node_data_from_dictionary=None, plot_label_title=None):
        """
            Visualiza um nó específico ou um conjunto de nós no grafo.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        # Se for passado informação dos nós por um dicionário usar para fazer o plot
        if node_data_from_dictionary:
            graph_data = node_data_from_dictionary
        elif node:
            # Caso contrário apenas foi passado o id do node (nome do sender) usar apenas a informação dele
            graph_data = {node: self._graph[node]}
        else:
            raise Exception(
                "Não foi passado qualquer fonte de dados - 'node' ou 'node_data_from_dictionary' - na função visualize_node")

        # Cria o dígrafo
        G = nx.DiGraph()

        # Adiciona os nós e arestas ap grafo
        for node, data in graph_data.items():
            G.add_node(node)
            for receiver, amount in data['sent_to'].items():
                G.add_edge(node, receiver, weight=amount)
            for sender, amount in data['received_from'].items():
                G.add_edge(sender, node, weight=amount)

        # Adiciona cor aos nós
        color_map = []
        for node in G:
            if node in graph_data.keys():
                color_map.append('red')
            else:
                color_map.append('skyblue')

        # https://stackoverflow.com/questions/56409016/networkx-networks-with-many-nodes
        # print([x for x in nx.__dir__() if x.endswith('_layout')])
        # ['bipartite_layout', 'circular_layout', 'kamada_kawai_layout', 'random_layout', 'rescale_layout', 'shell_layout', 'spring_layout', 'spectral_layout', 'planar_layout', 'fruchterman_reingold_layout', 'spiral_layout', 'multipartite_layout', 'bfs_layout', 'arf_layout']
        # pos = nx.spring_layout(G)
        pos = nx.arf_layout(G)

        # Plot the graph
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
        # fix que permitiu mostrar o peso ambos os nós ligados entre si ( A->B e B->A ):
        #   - https://stackoverflow.com/questions/66044075/weight-of-edges-is-not-showing
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), label_pos=0.33)
        nx.draw(G, pos, with_labels=True,
                node_size=5000,
                node_color=color_map,
                font_size=10,
                font_weight='bold',
                arrows=True,
                connectionstyle='arc3, rad = 0.1')

        plt.title(f"Grafo Orientado que representa as Transações {plot_label_title}")
        plt.show()

    def top_transacting_accounts(self):
        """
            Obtém as 3 contas com o maior número de transações realizadas.
        """
        top_3_transacting_accounts = sorted(self._graph.items(),
                                            key=lambda x: sum(x[1]['number_of_transactions_sent_to'].values()),
                                            reverse=True)[0:3]

        # DEBUG ---------------------------------------------------------------------------------
        # print("chamado do interior do método 'top_transacting_accounts':")
        # pprint.pprint(top_3_transacting_accounts) # mostra a lista, pois em lista mantém a ordem

        for each_account in top_3_transacting_accounts:
            # print(each_account, "com", len(each_account[1][sent_to]), "transações")
            print(each_account[0], end=" - ")
            amount_of_transactions = lambda x: sum(x)
            print(
                amount_of_transactions(
                    self._graph[each_account[0]]['number_of_transactions_sent_to'].values()
                ),
                "transações (de envio) totais"
            )
        #----------------------------------------------------------------------------------------

        # se for para retornar um simplesmente a lista: -------
        # return top_3_transacting_accounts
        # -----------------------------------------------------

        # se for para retornar um dicionário: ----------------------------------------------------------------
        # Nota: ao converter para dicionário, por questões internas de eficiência do python, perdemos a ordem
        # return dict(top_3_transacting_accounts)
        # ----------------------------------------------------------------------------------------------------

        # se for para retornar um plot do grafo dessas contas: --------------------------------------------------------------------------
        self.visualize_node(node_data_from_dictionary=dict(top_3_transacting_accounts),
                            plot_label_title=" - Top 3 das Contas com mais Transações realizadas")
        # -------------------------------------------------------------------------------------------------------------------------------

    def top_volume_accounts(self):
        """
            Obtém as 3 contas que transacionaram maior volume (euros)
        """

        # Calculate total volume sent by each node
        node_volumes = {}
        for node, info in self._graph.items():
            total_sent = sum(info['sent_to'].values())
            node_volumes[node] = total_sent

        # Sort nodes by total volume sent
        sorted_nodes = sorted(node_volumes.items(), key=lambda x: x[1], reverse=True)

        # DEBUG -----------------------------------------------------------
        print("chamado do interior do método 'top_volume_accounts':")
        print("sorted_nodes[0:3]:")
        pprint.pprint(sorted_nodes[0:3])

        # obter o top 3 de nós com maior total de volume enviado
        top_3_volume_accounts = {node: self._graph[node] for node, _ in sorted_nodes[0:3]}

        # DEBUG -----------------------------------------------------------
        # print("chamado do interior do método 'top_volume_accounts':")
        # pprint.pprint(top_3_volume_accounts)

        # se for para retornar um dicionário: -----------------
        # return top_3_volume_accounts
        # -----------------------------------------------------

        # se for para retornar um plot do grafo dessas contas: ------------------------------------------------------------------
        self.visualize_node(node_data_from_dictionary=top_3_volume_accounts,
                            plot_label_title=" - Top 3 das Contas que transacionaram (envio) maior Volume em Euros")
        # -----------------------------------------------------------------------------------------------------------------------

    def most_freq_txs(self):
        """
            Obtém os 3 pares de nós que efectuaram o maior número de transações entre si.
        """
        list_of_pairs_with_transactions_between_them = list()

        for node in self._graph.keys():

            # Verifica se o nó tem transações enviadas e recebidas
            if len(self._graph[node]["sent_to"]) > 0 and \
                    len(self._graph[node]["received_from"]) > 0:

                # DEBUG: ------------------------------------------------------------------
                # print("node:", node, "tem transações enviadas e transações recebidas")
                # -------------------------------------------------------------------------

                for sent_to_person in self._graph[node]["sent_to"].keys():

                    # Verifica se a pessoa que recebeu a transação também enviou uma transação para o nó atual
                    if sent_to_person in self._graph[node]["received_from"]:
                        # DEBUG: ----------------------------------------------------------------------------------------------
                        print("node:", node, "tem transações enviadas e transações recebidas com a pessoa",
                              sent_to_person)
                        # -----------------------------------------------------------------------------------------------------

                        total_number_of_transactions_between_them = self._graph[node]["number_of_transactions_sent_to"][
                                                                        sent_to_person] + \
                                                                    self._graph[node][
                                                                        "number_of_transactions_received_from"][
                                                                        sent_to_person]

                        # DEBUG:
                        # print("total de transações (de envio e de recepção somadas) realizadas entre os dois:",
                        #      total_number_of_transactions_between_them)

                        list_of_pairs_with_transactions_between_them.append(
                            [node, sent_to_person, total_number_of_transactions_between_them]
                        )

        # ordena pelo critério do nº total de transações entre os dois intervenientes
        list_of_pairs_with_transactions_between_them = sorted(list_of_pairs_with_transactions_between_them,
                                                              key=lambda x: x[2], reverse=True)

        # Remover os pares repetidos (pois não interessa a ordem)
        for each_pair in list_of_pairs_with_transactions_between_them:
            for repeated_pair in list_of_pairs_with_transactions_between_them:
                if each_pair[0] == repeated_pair[1] and each_pair[1] == repeated_pair[0]:
                    list_of_pairs_with_transactions_between_them.remove(repeated_pair)

        # filtrar os 3 maiores -----------------------------------------------------------------------------
        list_of_pairs_with_transactions_between_them = list_of_pairs_with_transactions_between_them[0:3]
        # --------------------------------------------------------------------------------------------------

        # DEBUG: --------------------------------------------------------
        print("list_of_pairs_with_transactions_between_them[0:3]:")
        pprint.pprint(list_of_pairs_with_transactions_between_them)
        # ---------------------------------------------------------------

        # Armazena a informação, presente no grafo, de cada um dos nós
        dict_of_nodes_with_most_freq_txs = {}

        for info_tuple in list_of_pairs_with_transactions_between_them:
            node1, node2, _ = info_tuple
            if node1 not in dict_of_nodes_with_most_freq_txs:
                dict_of_nodes_with_most_freq_txs[node1] = self._graph[node1]

            if node2 not in dict_of_nodes_with_most_freq_txs:
                dict_of_nodes_with_most_freq_txs[node2] = self._graph[node2]

        # DEBUG: --------------------------------------------------------
        # print("dict_of_nodes_with_most_freq_txs:")
        # pprint.pprint(dict_of_nodes_with_most_freq_txs)
        # ---------------------------------------------------------------

        # se for para retornar um dicionário: -----------------
        # return dict_of_nodes_with_most_freq_txs
        # -----------------------------------------------------

        # se for para retornar um plot do grafo dessas contas: ------------------------------------------------------------------------------------------------
        self.visualize_node(node_data_from_dictionary=dict_of_nodes_with_most_freq_txs,
                            plot_label_title=" - Top 3 dos Pares de nós com mais Transações entre Si")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------

    def highest_volume_txs(self):
        """
            Obtém as transações com maior volume
        """
        list_of_pairs_with_bigger_volume_between_them = list()

        for node in self._graph.keys():

            # Verifica se o nó tem transações enviadas e recebidas
            if len(self._graph[node]["sent_to"]) > 0 and \
                    len(self._graph[node]["received_from"]) > 0:

                # DEBUG: ------------------------------------------------------------------
                # print("node:", node, "tem transações enviadas e transações recebidas")
                # -------------------------------------------------------------------------

                for sent_to_person in self._graph[node]["sent_to"].keys():

                    # Verifica se a pessoa que recebeu a transação também enviou uma transação para o nó atual
                    if sent_to_person in self._graph[node]["received_from"]:
                        # DEBUG: ----------------------------------------------------------------------------------------------
                        print("node:", node, "tem transações enviadas e transações recebidas com a pessoa",
                              sent_to_person)
                        # -----------------------------------------------------------------------------------------------------

                        total_volume_between_them = self._graph[node]["sent_to"][sent_to_person] + \
                                                    self._graph[node]["received_from"][sent_to_person]

                        print("volume (euros) total realizado entre os dois:", total_volume_between_them)

                        list_of_pairs_with_bigger_volume_between_them.append(
                            [node, sent_to_person, total_volume_between_them]
                        )

        # sort by total number of transactions between them
        list_of_pairs_with_bigger_volume_between_them = sorted(list_of_pairs_with_bigger_volume_between_them,
                                                               key=lambda x: x[2], reverse=True)

        # Remove os pares repetidos
        for each_pair in list_of_pairs_with_bigger_volume_between_them:
            for repeated_pair in list_of_pairs_with_bigger_volume_between_them:
                if each_pair[0] == repeated_pair[1] and each_pair[1] == repeated_pair[0]:
                    list_of_pairs_with_bigger_volume_between_them.remove(repeated_pair)

        # filtrar os 3 maiores -----------------------------------------------------------------------------
        list_of_pairs_with_bigger_volume_between_them = list_of_pairs_with_bigger_volume_between_them[0:3]
        # --------------------------------------------------------------------------------------------------

        # DEBUG: --------------------------------------------------------
        print("list_of_pairs_with_bigger_volume_between_them[0:3]:")
        pprint.pprint(list_of_pairs_with_bigger_volume_between_them)
        # ---------------------------------------------------------------

        # Armazena a informação de cada um dos nós
        dict_of_nodes_with_bigger_volume_between_them = {}

        for info_tuple in list_of_pairs_with_bigger_volume_between_them:
            node1, node2, _ = info_tuple
            if node1 not in dict_of_nodes_with_bigger_volume_between_them:
                dict_of_nodes_with_bigger_volume_between_them[node1] = self._graph[node1]

            if node2 not in dict_of_nodes_with_bigger_volume_between_them:
                dict_of_nodes_with_bigger_volume_between_them[node2] = self._graph[node2]

        # DEBUG: --------------------------------------------------------
        print("dict_of_nodes_with_bigger_volume_between_them:")
        pprint.pprint(dict_of_nodes_with_bigger_volume_between_them)
        # ---------------------------------------------------------------

        # se for para retornar um dicionário: -----------------
        # return dict_of_nodes_with_bigger_volume_between_them
        # -----------------------------------------------------

        # se for para retornar um plot do grafo dessas contas: ------------------------------------------------------------------------------------------------
        self.visualize_node(node_data_from_dictionary=dict_of_nodes_with_bigger_volume_between_them,
                            plot_label_title=" - Top 3 dos Pares de nós com mais Volume (euros) transacionado entre Si")
        # -----------------------------------------------------------------------------------------------------------------------------------------------------

    def get_two_random_nodes_from_graph(self):
        """
            Obtém dois nós aleatórios que constam no grafo
        """
        import random
        return random.sample(list(self._graph.keys()), 2)

    # ----------------------------------------------------------------------------------------------------------------

    # Métodos para a parte final (pág 7) do enunciado ----------------------------------------------------------------
    def dijkstra_max_path(self, graph, start, end):
        """
            Encontra o caminho mais pesado no grafo.
        """
        queue = [(start, [], 0)]  # (nó atual, caminho até o nó atual, peso acumulado)
        max_path = []
        max_weight = 0
        visited = set()  # Conjunto dos nós visitados

        while queue:
            node, path, weight = queue.pop(0)

            if node in path:
                continue  # Evita ciclos

            path = path + [node]

            if node == end:
                # Se o peso acumulado for maior que o maior peso encontrado até agora
                if weight > max_weight:
                    # Atualiza o caminho máximo e o maior peso
                    max_path = path
                    max_weight = weight
                continue

            visited.add(node)

            # Itera sobre as out edges do nó atual
            for edge in self.get_edges(node, out=True):

                neighbor = edge.get_nodes()[1]  # Obtém o nó vizinho
                if neighbor not in visited:
                    # Calcula o peso total do caminho até ao vizinho
                    total_weight = weight + edge.get_info()
                    # Adiciona o vizinho à fila com o novo caminho e peso acumulado
                    queue.append((neighbor, path, total_weight))

        return max_weight, max_path

    # Desenhar o grafo com o caminho mais pesado destacado
    def draw_graph_with_path(self, path):
        """
            Desenha todo o grafo destacando o caminho passado como argumento.
        """
        import pandas as pd
        import networkx as nx
        import matplotlib.pyplot as plt
        from io import StringIO

        df = pd.read_csv("transactions.csv")

        # Construir o grafo
        G = nx.DiGraph()
        for _, row in df.iterrows():
            G.add_edge(row['sender'], row['receiver'], weight=row['value'])

        pos = nx.arf_layout(G)

        # Define as cores das arestas: verde para as arestas que fazem parte do caminho e preto para as restantes
        edge_colors = ['green' if (u, v) in zip(path, path[1:]) else 'black' for u, v in G.edges()]
        # Define as larguras das arestas: 4 para as que fazem parte do caminho e 1 para as restantes
        edge_widths = [4 if (u, v) in zip(path, path[1:]) else 1 for u, v in G.edges()]
        # Define as cores dos nós: vermelho para os nós que fazem parte do caminho e azul claro para os restantes
        node_colors = ['red' if node in path else 'skyblue' for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)

        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), label_pos=0.33)

        nx.draw(G, pos, with_labels=True,
                node_color=node_colors,
                edge_color=edge_colors,
                width=edge_widths,
                node_size=800,
                font_size=6,
                font_weight='bold',
                arrows=True,
                connectionstyle='arc3, rad = 0.1')

        plt.title(f'Grafo Orientado com o "caminho com as ligações mais fortes" entre {path[0]} e {path[-1]}')
        plt.show()
    # ----------------------------------------------------------------------------------------------------------------
