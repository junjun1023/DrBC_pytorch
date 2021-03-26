import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.typing import Adj, OptTensor

import torch_geometric.utils as utils

import os
import json
import numpy as np

class Encoder(MessagePassing):
        def __init__(self, c, p, num_layers, device, aggr="add"):
                super(Encoder, self).__init__(aggr=aggr)
                
                self.num_layers = num_layers
                self.w_0 = nn.Linear(in_features=c, out_features=p).double()
                
                self.relu = nn.ReLU(inplace=True)
                self.rnn = nn.GRUCell(p, p).double()

                self.device = device

        def forward(self, data):
                
                x, edge_index = data.x, data.edge_index
                
                # compute dgree
                row, col = edge_index
                deg = utils.degree(col)
                deg = torch.add(deg, 1)
                deg_inv_sqrt = torch.pow(deg, -0.5)
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                
                
                # h_0 = x

                # h_1
                x = self.w_0(x)
                x = self.relu(x)
                x = F.normalize(x, p=2, dim=1)
                
                h_s = [x]
                
                for i in range(self.num_layers-1):
                        # internally calls the message(), aggregate() and update() functions
                        x = self.propagate(edge_index, x=x, norm=norm)
                        h_s.append(x)
                
                h_s = torch.stack(h_s, dim=-1)
                # Use torch.max to do max_pooling
                z, _ = torch.max(h_s, dim=-1)
                
                return z

        def message(self, x_j, norm: OptTensor):
                """      
                In addition, tensors passed to propagate() can be mapped to the respective nodes i and j 
                by appending _i or _j to the variable name, .e.g. x_i and x_j. 
                Note that we generally refer to i as the central nodes that aggregates information, 
                and refer to j as the neighboring nodes, since this is the most common notation.
                """
        
                return x_j if norm is None else norm.view(-1, 1) * x_j
        
        
        def update(self, aggr_out, x):
                """        
                Takes in the output of aggregation as first argument 
                and any argument which was initially passed to propagate().
                """

                x = self.rnn(x, aggr_out)
                x = F.normalize(x, p=2, dim=1) 
                
                return x
        

class Decoder(nn.Module):
        def __init__(self, p, q):
                
                super().__init__()
                
                self.w_4 = nn.Linear(in_features=p, out_features=q).double()
                self.w_5 = nn.Linear(in_features=q, out_features=1).double()
                
                self.relu = nn.ReLU(inplace=True)
                
        def forward(self, z):
                z = self.w_4(z)
                z = self.relu(z)
                z = self.w_5(z)
                
                return z


class DrBC(nn.Module):
        def __init__(self, encoder_params, decoder_params):
                super().__init__()
                
                self.encoder = Encoder(**encoder_params)
                self.decoder = Decoder(**decoder_params)
                
                
        def forward(self, data):
                
                z = self.encoder(data)
                
                return self.decoder(z)


def load_checkpoint(filepath, device, **params):
        model = DrBC(**params["drbc"])
        model = model.to(device)
        
        if os.path.exists(filepath):
                print("pretrained finded")
                checkpoint = torch.load(filepath)
                model.load_state_dict(checkpoint['model_stat'])
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                optimizer.load_state_dict(checkpoint['optimizer_stat'])

        else:
                print("use a new optimizer")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        return model, optimizer


def visualize_plt(train_p, valid_p, save_p):
        import matplotlib.pyplot as plt

        with open(train_p, 'r') as train_info, \
                open(valid_p, 'r') as valid_info:
        
                train = json.load(train_info)
                valid = json.load(valid_info)

                keys = list(train.keys())
                epochs = len(train[keys[0]])

                x = np.linspace(1, epochs, epochs)
                
                ### plot bce loss and mean top N %
                fig, axs = plt.subplots(len(keys), figsize=(25, 4 * len(keys)))
                for index, ax in enumerate(axs):
                        key = keys[index]
                        
                        ax.plot(x, train[key], color="blue")
                        ax.plot(x, valid[key], color="orange")
                        
                        ax.legend(["train", "valid"], loc='upper left')
                        ax.set_title(key)

                        ax.grid()

                        if key=="bce" or key=="mse":
                                pass
                        else:
                                ax.set_ylim([0, 1])
                        
                plt.tight_layout()
                plt.savefig(save_p, facecolor="white")