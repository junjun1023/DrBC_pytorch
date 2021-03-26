from utils.eval import kendal_tau_distance, top_n_percentage
from utils.data import from_networkx
import networkx as nx
import networkit as nk
import numpy as np
import time

def run_kadabra(edge_lists, scores, x_sep, y_sep, usecols):

        top1_list = []
        top5_list = []
        top10_list = []
        kendal_list = []
        time_list = []

        for edge_list, score in zip(edge_lists, scores):
                G_nx = nx.readwrite.edgelist.read_edgelist(edge_list, delimiter=x_sep)
                G_nk = nk.nxadapter.nx2nk(G_nx)

                gt = np.loadtxt(score, delimiter=y_sep, usecols=usecols)
                
                method = nk.centrality.KadabraBetweenness(G_nk, 0.05, 0.8)
                start = time.time()
                method.run()
                end = time.time()
                
                top1_list.append(top_n_percentage(np.array(method.scores()),  gt, k=1, device="cpu"))
                top5_list.append(top_n_percentage(np.array(method.scores()), gt, k=5, device="cpu"))
                top10_list.append(top_n_percentage(np.array(method.scores()), gt, k=10, device="cpu"))
                kendal_list.append(kendal_tau_distance(np.array(method.scores()), gt))
                time_list.append(end-start)

        return top1_list, top5_list, top10_list, kendal_list, time_list


def run_rk(edge_lists, scores, x_sep, y_sep, usecols):

        top1_list = []
        top5_list = []
        top10_list = []
        kendal_list = []
        time_list = []

        for edge_list, score in zip(edge_lists, scores):
                G_nx = nx.readwrite.edgelist.read_edgelist(edge_list, delimiter=x_sep)
                G_nk = nk.nxadapter.nx2nk(G_nx)

                gt = np.loadtxt(score, delimiter=y_sep, usecols=usecols)
                
                method = nk.centrality.ApproxBetweenness(G_nk, epsilon=0.1)
                start = time.time()
                method.run()
                end = time.time()
                
                top1_list.append(top_n_percentage(np.array(method.scores()), gt, k=1, device="cpu"))
                top5_list.append(top_n_percentage(np.array(method.scores()), gt, k=5, device="cpu"))
                top10_list.append(top_n_percentage(np.array(method.scores()), gt, k=10, device="cpu"))
                kendal_list.append(kendal_tau_distance(np.array(method.scores()), gt))
                time_list.append(end-start)

        return top1_list, top5_list, top10_list, kendal_list, time_list


def run_kbc(edge_lists, scores, x_sep, y_sep, usecols):
        import os
        import subprocess

        top1_list = []
        top5_list = []
        top10_list = []
        kendal_list = []
        time_list = []

        for edge_list, score in zip(edge_lists, scores):

                base = os.path.splitext(os.path.basename(edge_list))[0]
                        
                G_nx = nx.readwrite.edgelist.read_edgelist(edge_list, delimiter=x_sep)
                G_pyg = from_networkx(G_nx)
                
                arr = np.array([G_pyg.x.shape[0], G_pyg.edge_index.shape[1]]).reshape((1, 2))
                arr = np.concatenate([arr, G_pyg.edge_index.t().numpy()])

                save = os.path.join("BeBeCA/Source_Code/5000", "{}.txt".format(base))
                save_pr = os.path.join("BeBeCA/Source_Code/5000", "{}_pr.txt".format(base))
                
                np.savetxt(save, arr, fmt="%d")
                
                start = time.time()
                subprocess.run(["./BeBeCA/Source_Code/KPATH", "2", save, save_pr])
                end = time.time()
                
                pr = np.loadtxt(save_pr, delimiter=y_sep, usecols=usecols)
                gt = np.loadtxt(score, usecols=1)
                
                top1_list.append(top_n_percentage(pr, gt, k=1, device="cpu"))
                top5_list.append(top_n_percentage(pr, gt, k=5, device="cpu"))
                top10_list.append(top_n_percentage(pr, gt, k=10, device="cpu"))
                kendal_list.append(kendal_tau_distance(pr, gt))
                time_list.append(end-start)

        return top1_list, top5_list, top10_list, kendal_list, time_list