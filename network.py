
import igraph
from igraph import *
from nltk import bigrams
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy import integrate

def get_largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

class CNetwork(object):

    def __init__(self, document, model, percentages):
        self.document = document
        self.model = model
        self.percentages = percentages
        self.words = list(set(self.document))
        self.word_dict = {word:index for index, word in enumerate(self.words)}

    def create_network(self):
        edges = []
        string_bigrams = bigrams(self.document)
        for i in string_bigrams:
            edges.append((i[0], i[1]))
            #edges.append((self.word_dict[i[0]], self.word_dict[i[1]]))

        network = Graph()
        network.add_vertices(self.words)
        #network.add_vertices(len(self.words))
        network.add_edges(edges)
        network.simplify()
        print('Nodes:', len(self.words), '-', 'Edges:', len(network.get_edgelist()))
        return network

    def add_embeddings(self, network):
        network_size = network.vcount()
        actual_edges = network.get_edgelist()
        num_edges = network.ecount()
        maximum_num_edges = int((network_size * (network_size - 1)) / 2)
        remaining_edges = maximum_num_edges - num_edges
        print('Testing available edges:', maximum_num_edges, remaining_edges)
        edges_to_add = []

        for percentage in self.percentages:
            value = int(num_edges * percentage / 100) + 1
            edges_to_add.append(value)
        #print(edges_to_add)
        #words = list(set(self.document))
        matrix = []
        for word in self.words:
            embedding = self.model[word]
            matrix.append(embedding)

        matrix = np.array(matrix)
        similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        similarity_matrix[np.triu_indices(network_size)] = -1
        similarity_matrix[similarity_matrix == 1.0] = -1
        largest_indices = get_largest_indices(similarity_matrix, maximum_num_edges)

        max_value = np.max(edges_to_add)
        counter = 0
        index = 0
        new_edges = []
        while counter < max_value:
            x = largest_indices[0][index]
            y = largest_indices[1][index]
            if not network.are_connected(x, y):
                new_edges.append((x, y))
                counter += 1
            index += 1

        networks = []
        for value in edges_to_add:
            edges = []
            edges.extend(actual_edges)
            edges.extend(new_edges[0:value])
            new_network = Graph()
            new_network.add_vertices(self.words)
            new_network.add_edges(edges)
            networks.append(new_network)
        return networks

    def create_networks(self):
        network = self.create_network()
        networks = self.add_embeddings(network)
        networks.insert(0, network)

        prueba = [len(net.get_edgelist()) for net in networks]
        print('Num edges in networks:', prueba)
        return networks

    def get_weighted_network(self, words):
        matrix = []
        for word in words:
            embedding = self.model[word]
            matrix.append(embedding)
        matrix = np.array(matrix)
        similarity_matrix = 1 - pairwise_distances(matrix, metric='cosine')
        simList = similarity_matrix.tolist()
        return Graph.Weighted_Adjacency(simList, mode="undirected", attr="weight", loops=False)

    def get_alpha(self, k, p_ij):
        try:
            alpha = 1 - (k - 1) * integrate.quad(lambda x: (1 - x) ** (k - 2), 0, p_ij)[0]
        except:
            alpha = 1
        return alpha

    def disparity_filter(self, network):
        print('Calculating disparity filter')
        degree = network.degree()
        for vertex in range(network.vcount()):
            k = degree[vertex]
            neighbors = network.neighbors(vertex)
            if k > 1:
                sum_w = network.strength(vertex, weights=network.es['weight'])
                for v in neighbors:
                    w = network.es[network.get_eid(vertex, v)]['weight']
                    p_ij = float(np.absolute(w)) / sum_w
                    alpha_ij = self.get_alpha(k, p_ij)
                    network.es[network.get_eid(vertex, v)]['alpha'] = alpha_ij
            else:
                network.es[network.get_eid(vertex, neighbors[0])]['alpha'] = 0

    def add_filtered_embeddings(self, network):
        print('testing ...')
        weighted_network = self.get_weighted_network(self.words)
        #print('len edges', len(weighted_network.get_edgelist()))
        self.disparity_filter(weighted_network)
        #alphas = weighted_network.es['alpha']
        #print('len edges', len(weighted_network.get_edgelist()), len(alphas))

        extra_edges = []
        all_edges = weighted_network.es
        print('Looking for embedding edges')
        for edge_values in all_edges:
            edge = edge_values.tuple
            if not network.are_connected(edge[0], edge[1]):
                extra_edges.append(edge_values)

        sorted_extra_edges = sorted(extra_edges, key=lambda x: x["alpha"], reverse=False)
        sorted_extra_edges = sorted_extra_edges[0:len(network.get_edgelist())]

        worst_edges =  sorted_extra_edges[::-1]

        num_edges = len(sorted_extra_edges)#*2
        top_k_to_remove = []

        for percentage in self.percentages:
            top_k = int(num_edges * percentage / 100) + 1
            top_k_to_remove.append(top_k)

        auxiliar_network = Graph()
        auxiliar_network.add_vertices(self.words)
        edges = [(e.source, e.target) for e in sorted_extra_edges]
        auxiliar_network.add_edges(network.get_edgelist())
        auxiliar_network.add_edges(edges)

        print('Num edges:', num_edges)
        print('top k remove:',top_k_to_remove)
        print('worst edges:', len(worst_edges))
        filtered_networks = []
        for top in top_k_to_remove:
            remove = worst_edges[0:top]
            r_edges = [(e.source, e.target) for e in remove]
            new_network = auxiliar_network.copy()
            new_network.delete_edges(r_edges)
            filtered_networks.append(new_network)
        return filtered_networks


        '''
        filtered_networks = []
        for top in top_k_to_remove:
            remove = sorted_extra_edges[0:top]
            edges_to_remove = [(e.source, e.target) for e in remove]
            #auxiliar_network = weighted_network.copy()
            new_network = Graph()
            new_network.add_vertices(self.words)
            new_network.add_edges(weighted_network.get_edgelist())
            new_network.delete_edges(edges_to_remove)
            #auxiliar_network =
            #auxiliar_network.delete_edges(edges_to_remove)
            #print(len(auxiliar_network.get_edgelist()), len(edges_to_remove))
            filtered_networks.append(new_network)
        return filtered_networks
        '''


    def create_filtered_networks(self):
        network = self.create_network()
        networks = self.add_filtered_embeddings(network)
        networks.insert(0, network)

        prueba = [len(net.get_edgelist()) for net in networks]
        print('Num edges in networks:', prueba)
        a = input()
        return networks




    def get_frequency_counts(self, networks, features):
        found_features = []
        for word in features:
            try:
                node = networks[0].vs.find(name=word)
            except:
                node = None
            if node is not None:
                found_features.append(word)
        network_features = []
        for network in networks:
            dgr = network.degree(found_features)
            feature = [0.0 for _ in range(len(features))]
            for word, value in zip(found_features, dgr):
                feature[features[word]] = value
            #network_features.extend(feature)
            network_features.append(feature)
        network_features = np.array(network_features)
        return network_features