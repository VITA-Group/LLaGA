import math

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import numpy as np
import pandas as pd
import torch
import random
# import torch_geometric.transforms as T
from sentence_transformers import SentenceTransformer
import os
from tqdm import trange
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import openai
from multiprocessing import Process
import scipy.sparse as sp
# from multiprocessing import  pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, remove_self_loops, add_self_loops
import copy
import json
from tqdm import tqdm
from utils.constants import DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_TOKEN

"""
this function is for getting node sequence around  mode [node_idx], use avoid_idx for link prediction task to filter the other node
"""
def get_fix_shape_subgraph_sequence_fast(edge_list, node_idx, k_hop, sample_size, avoid_idx=None):
    assert k_hop > 0 and sample_size > 0
    neighbors = [[node_idx]]
    for t in range(k_hop):
        last_hop = neighbors[-1]
        current_hop = []
        for i in last_hop:
            if i == DEFAULT_GRAPH_PAD_ID:
                current_hop.extend([DEFAULT_GRAPH_PAD_ID]*sample_size)
                continue
            node_neighbor = copy.copy(edge_list[i])
            if t == 0 and avoid_idx is not None and  avoid_idx in node_neighbor:
                node_neighbor.remove(avoid_idx)
            if len(node_neighbor) > sample_size:
                sampled_neighbor = random.sample(node_neighbor, sample_size)
            else:
                sampled_neighbor = node_neighbor + [DEFAULT_GRAPH_PAD_ID] * (sample_size - len(node_neighbor))
            current_hop.extend(sampled_neighbor)
        neighbors.append(current_hop)
    node_sequence = [n for hop in neighbors for n in hop]
    return node_sequence

"""
get edge_list from pyg edge_index\
"""
def generate_edge_list(data):
    # data = torch.load(os.path.join(data_dir, "processed_data.pt"))
    row, col = data.edge_index
    n = data.num_nodes
    edge_list= [[] for _ in range(n)]
    row=row.numpy()
    col=col.numpy()

    for i in trange(row.shape[0]):
        edge_list[row[i]].append(int(col[i]))
    # torch.save(edge_list, os.path.join(data_dir, "edge_list.pt"))
    return edge_list

from torch_geometric.utils import k_hop_subgraph
class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).

    def partition_propagate(self, data_edge_index, x, norm, select_idx=None, chunk_size=800, cuda=False):
        if select_idx is None:
            n = x.shape[0]
            select_idx = torch.arange(n)
        else:
            n = select_idx.shape[0]

        os=[]
        for i in trange(0, n, chunk_size):
            key=select_idx[i:i+chunk_size]
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(key, 1, data_edge_index, relabel_nodes=True)
            if cuda:
                o =  self.propagate(edge_index.cuda(), x=x[subset].cuda(), norm=norm[edge_mask].cuda())
            else:
                o = self.propagate(edge_index, x=x[subset], norm=norm[edge_mask])
            os.append(o[mapping])

        return torch.cat(os, dim=0)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


def generate_multi_hop_x_arxiv(emb="sbert"):
    data = torch.load(f"dataset/ogbn-arxiv/processed_data_link_notest.pt")
    x = torch.load(f"dataset/ogbn-arxiv/{emb}_x.pt")
    edge_index = data.edge_index
    row, col = data.edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    link_test_path = os.path.join(f"dataset/ogbn-arxiv", "edge_sampled_2_10_only_test.jsonl")
    with open(link_test_path, 'r') as f:
        link_test_lines = f.readlines()
    link_test_lines = [json.loads(line) for line in link_test_lines]
    n = data.num_nodes
    mask = torch.full([n], fill_value=False, dtype=torch.bool)
    for link in link_test_lines:
        mask[link['id'][0]] = True
        mask[link['id'][1]] = True
    mp = MP()
    torch.save(x, f"dataset/ogbn-arxiv/{emb}_0hop_x.pt")
    for i in range(4):
        x = mp.propagate(edge_index, x=x, norm=norm)
        torch.save(x[mask].cpu(), f"dataset/ogbn-arxiv/{emb}_{i + 1}hop_x_notestlink.pt")



def generate_multi_hop_x_products(emb="sbert"):
    print(emb)
    data = torch.load(f"dataset/ogbn-products/processed_data_link_notest.pt")
    x = torch.load(f"dataset/ogbn-products/{emb}_x.pt")
    row, col = data.edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    link_test_path = os.path.join(f"dataset/ogbn-products", "edge_sampled_2_10_only_test.jsonl")
    with open(link_test_path, 'r') as f:
        link_test_lines = f.readlines()
    link_test_lines = [json.loads(line) for line in link_test_lines]
    n = data.num_nodes
    mask = torch.full([n], fill_value=False, dtype=torch.bool)
    for link in link_test_lines:
        mask[link['id'][0]] = True
        mask[link['id'][1]] = True
    mp = MP()
    torch.save(x, f"dataset/ogbn-products/{emb}_0hop_x.pt")
    for i in range(4):
        x = mp.partition_propagate(data.edge_index, x=x, norm=norm, chunk_size=200, cuda=True)
        torch.save(x[mask].cpu(), f"dataset/ogbn-products/{emb}_{i + 1}hop_x_notestlink.pt")