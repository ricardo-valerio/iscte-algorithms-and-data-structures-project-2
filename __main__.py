# Este if permite-nos executar código quando o ficheiro é executado como um script,
# isto é, quando é chamado pelo interpretador mas não quando é importado como um módulo.
# referência: https://realpython.com/if-name-main-python/
if __name__ == '__main__':
    # importar tudo do ficheiro graph.py, incluíndo os módulos que são importados nele.
    from graph import *

    # Criar uma instância da classe BankTransfersGraph
    bank_transactions_graph = BankTransfersGraph()

    # Carregar o ficheiro e preencher o grafo com a informação nele contida
    bank_transactions_graph.load_file(file_path="transactions.csv")

    # mostrar o grafo todo
    pprint.pprint(bank_transactions_graph._graph, width=1)

    print("\n\n=================================================================\n\n")

    print("nº de nós no grafo:", len(bank_transactions_graph._graph.keys()))
    print("outra forma de ver o nº de nós no grafo:", bank_transactions_graph.nr_nodes())

    print("nº de arestas/edges no grafo:", bank_transactions_graph.nr_edges())

    print("nº total de transações:", bank_transactions_graph.nr_transactions())

    print("\n\n=================================================================\n\n")

    for node in bank_transactions_graph._graph.keys():
        print("in-degree do node:", node, bank_transactions_graph.in_degree(node))
        print("out-degree do node:", node, bank_transactions_graph.out_degree(node))

    print("\n\n=================================================================\n\n")

    print("most_connected_node:", bank_transactions_graph.most_connected_node())
    print("least_connected_node:", bank_transactions_graph.least_connected_node())

    print("\n\n=================================================================\n\n")

    print(f"neighbours('{bank_transactions_graph.most_connected_node()}')",
          bank_transactions_graph.neighbours(node=bank_transactions_graph.most_connected_node(), out=False))

    print(f"neighbours('{bank_transactions_graph.most_connected_node()}')",
          bank_transactions_graph.neighbours(node=bank_transactions_graph.most_connected_node(), out=True))

    print("\n\n=================================================================\n\n")

    print("edge_with_highest_weight:", bank_transactions_graph.edge_with_highest_weight())
    print("edge_with_lowest_weight:", bank_transactions_graph.edge_with_lowest_weight())

    print("\n\n=================================================================\n\n")

    # Testar plot do grafo todo, isto é, com as transações todas: -------------------------
    print("A apresentar a janela do plot do grafo...")
    bank_transactions_graph.plot_graph()
    print("A janela do plot foi fechada...")
    # -------------------------------------------------------------------------------------

    print("\n\n=================================================================\n\n")

    print("avg_edges:", bank_transactions_graph.avg_edges())

    print("\n\n=================================================================\n\n")

    print("Out edges do 'most_connected_node':")
    pprint.pprint(bank_transactions_graph.get_edges(node=bank_transactions_graph.most_connected_node(), out=True))

    print("In edges do 'most_connected_node':")
    pprint.pprint(bank_transactions_graph.get_edges(node=bank_transactions_graph.most_connected_node(), out=False))

    print("\n\n=================================================================\n\n")

    print("A apresentar a janela do plot do grafo...")
    bank_transactions_graph.visualize_node(
            node=bank_transactions_graph.most_connected_node(),
            plot_label_title=" - Nó mais Conectado"
    )
    print("A janela do plot foi fechada...")

    print("\n\n=================================================================\n\n")

    print("A apresentar a janela do plot do grafo...")
    bank_transactions_graph.top_transacting_accounts()
    print("A janela do plot foi fechada...")

    print("\n\n=================================================================\n\n")

    print("A apresentar a janela do plot do grafo...")
    bank_transactions_graph.top_volume_accounts()
    print("A janela do plot foi fechada...")

    print("\n\n=================================================================\n\n")

    print("A apresentar a janela do plot do grafo...")
    bank_transactions_graph.most_freq_txs()
    print("A janela do plot foi fechada...")

    print("\n\n=================================================================\n\n")

    print("A apresentar a janela do plot do grafo...")
    bank_transactions_graph.highest_volume_txs()
    print("A janela do plot foi fechada...")

    print("\n\n=================================================================\n\n")

    # Encontrar o caminho mais pesado entre dois nós aleatórios do grafo -----------------------------------------------------
    start_node, end_node = bank_transactions_graph.get_two_random_nodes_from_graph()
    max_weight, max_path = bank_transactions_graph.dijkstra_max_path(bank_transactions_graph._graph, start_node, end_node)

    print(f"O caminho mais pesado de {start_node} para {end_node} é: {max_path} com peso total de {max_weight}")

    # Desenhar o grafo com o caminho mais pesado destacado
    print("A apresentar a janela do plot do grafo...")
    bank_transactions_graph.draw_graph_with_path(path=max_path)
    print("A janela do plot foi fechada...")
    #-------------------------------------------------------------------------------------------------------------------------

    print("\n\n============================ PROGRAMA TERMINADO ==================================\n\n")
