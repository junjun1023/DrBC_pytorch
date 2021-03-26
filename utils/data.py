import os

import torch

import numpy as np
import json
from tqdm import tqdm
from datetime import datetime
import time
import matplotlib.pyplot as plt

import torch_geometric as pyg
# from torch_geometric.data import Data, DataLoader
import torch_geometric.utils as pyg_utils

import networkit as nk
import networkx as nx


def load_data(path, mode="between"):

        assert mode=="between" or mode=="closeness", "Unknown centrality mode."

        edge_index = []
        centrality = []
        for f in os.listdir(os.path.join(path, "graph")):
                basename = os.path.splitext(f)[0]
                p = os.path.join(path, "graph", f)
                edge_index.append(p)

                if mode=="between":
                        p = os.path.join(path, mode,  "{}_score.txt".format(basename))
                        centrality.append(p)
                elif mode == "closeness":
                        p = os.path.join(path, mode,  "{}_cc.txt".format(basename))                       
                        centrality.append(p)

        return edge_index, centrality



def split_data(path, x, y, replace=False):
        from sklearn.model_selection import train_test_split
        if os.path.exists(path) and replace:
                pass
        else:
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
                X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15)

                split = {
                        "X_train": X_train,
                        "X_valid": X_valid,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_valid": y_valid,
                        "y_test": y_test
                }

                with open(path, 'w') as f:
                        json.dump(split, f)


def to_data(x, y=None, x_sep=None, y_sep=None, usecols=None):
        """
        Read files and return pyg.Data
        """
        if x_sep is not None:
                edge_index = pyg.io.read_txt_array(x, dtype=torch.long, sep=x_sep)
        else:
                edge_index = pyg.io.read_txt_array(x, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        edge_index = pyg_utils.to_undirected(edge_index)

        row, col = edge_index  
        deg = pyg_utils.degree(col) # must use col to get degree, why?
        deg = deg.numpy()  

        vertice = []
        for d in deg:
                vertice.append([d, 1, 1])
        vertice = np.array(vertice, dtype=np.float)
        vertice = torch.from_numpy(vertice)

        if y is not None:
                ### between centrality
                score = np.loadtxt(y, delimiter=y_sep, usecols=usecols)
                score = np.reshape(score, (-1, 1))
                score = torch.from_numpy(score)

                data = pyg.data.Data(x=vertice, edge_index=edge_index, y=score)
                
        else:
                data = pyg.data.Data(x=vertice, edge_index=edge_index)

        return data


def to_dataloader(x, y, batch, y_sep=None, usecols=None):
        """
        Read files and return pyg.Dataloader
        """
        data_list = []
        for x_, y_ in zip(x, y):
                data = to_data(x_, y_, y_sep=y_sep, usecols=usecols)
                data_list.append(data)

        loader =pyg.data.DataLoader(data_list, batch_size=batch)
        return loader


def from_networkx(G, score_list=None):
        """Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
        :class:`torch_geometric.data.Data` instance.

        Args:
                G (networkx.Graph or networkx.DiGraph): A networkx graph.
        """

        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        data = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
                for key, value in feat_dict.items():
                        data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
                for key, value in feat_dict.items():
                        data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

        for key, item in data.items():
                try:
                        data[key] = torch.tensor(item)
                except ValueError:
                        pass

        data['edge_index'] = edge_index.view(2, -1)
        data['x'] = torch.from_numpy(
                np.array( [ [G.degree[i], 1, 1] for i in G.nodes()], dtype=np.float ) )

        if score_list is not None:
                data['y'] = torch.from_numpy(
                        np.array( [ [b] for b in score_list ] , dtype=np.float) )
        data = pyg.data.Data.from_dict(data)
        data.num_nodes = G.number_of_nodes()

        return data


def generate_nx_graph(nodes_cnt):
        # Draw network G from distribution D (like the power-law model)
        G = nx.generators.random_graphs.powerlaw_cluster_graph(n=nodes_cnt, m=4, p=0.05)
        # Calculate each node’s exact BC value bv, ∀v ∈ V
        betweenness = nx.algorithms.centrality.betweenness_centrality(G)
        
        # Convert betweenness dict to list
        between = [v for k, v in sorted(betweenness.items(), key=lambda  item: int(item[0]), reverse=False)]
        bc = np.array(between)
        
        closeness = nx.algorithms.centrality.closeness_centrality(G)
        closeness = [v for k, v in sorted(closeness.items(), key=lambda item: int(item[0]), reverse=False)]
        cc = np.array(closeness)
        
        return G, bc, cc