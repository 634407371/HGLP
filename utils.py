import os
import math
import torch
import random
import numpy as np
import networkx as nx
import scipy.io as sio
import scipy.sparse as ssp
import torch_geometric as tg
from tqdm import tqdm
from torch_geometric.data import Data


def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_graph(graph_name):
    graph = sio.loadmat("./data/{}.mat".format(graph_name))
    return graph['net']


def sample_neg(net, test_ratio):
    net_triu = ssp.triu(net, k=1)
    row, col, _ = ssp.find(net_triu)

    perm = random.sample(list(range(len(row))), len(row))
    row, col = row[perm], col[perm]
    split = int(math.ceil(len(row) * (1 - test_ratio)))
    train_pos = np.array([row[:split], col[:split]], dtype=int)
    test_pos = np.array([row[split:], col[split:]], dtype=int)

    train_num, test_num = train_pos.shape[1], test_pos.shape[1]
    neg = ([], [])
    n = net.shape[0]
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg = np.array([neg[0][:train_num], neg[1][:train_num]], dtype=int)
    test_neg = np.array([neg[0][train_num:], neg[1][train_num:]], dtype=int)
    return train_pos, train_neg, test_pos, test_neg


def get_train_graph(graph, test_pos):
    graph_copy = graph.copy()
    graph_copy[test_pos[0], test_pos[1]] = 0
    graph_copy[test_pos[1], test_pos[0]] = 0
    graph_copy.eliminate_zeros()
    return graph_copy


def links_to_subgraphs(graph, train_pos, train_neg, test_pos, test_neg, hop):
    train_pos_subgraphs, train_pos_max_num_node_labels = links2subgraphs(graph, train_pos, hop, 1)
    train_neg_subgraphs, train_neg_max_num_node_labels = links2subgraphs(graph, train_neg, hop, 0)
    test_pos_subgraphs, test_pos_max_num_node_labels = links2subgraphs(graph, test_pos, hop, 1)
    test_neg_subgraphs, test_neg_max_num_node_labels = links2subgraphs(graph, test_neg, hop, 0)
    train_subgraphs = train_pos_subgraphs + train_neg_subgraphs
    test_subgraphs = test_pos_subgraphs + test_neg_subgraphs
    max_num_node_labels = max(
        [train_pos_max_num_node_labels, train_neg_max_num_node_labels, test_pos_max_num_node_labels,
         test_neg_max_num_node_labels])
    return train_subgraphs, test_subgraphs, max_num_node_labels


def links2subgraphs(graph, links, hop, label):
    subgraph_list = []
    max_num_node_labels = 0
    for i, j in tqdm(zip(links[0], links[1])):
        subgraph, node_labels = subgraph_extraction_labeling((i, j), graph, hop)
        max_num_node_labels = max(max_num_node_labels, max(node_labels))
        node_labels = torch.tensor(node_labels, dtype=torch.long).unsqueeze(dim=0).T
        adj = nx.adjacency_matrix(subgraph)
        adj_triu = ssp.triu(adj)
        edge_index, _ = tg.utils.from_scipy_sparse_matrix(adj_triu)
        pyg_subgraph = Data(x=None, edge_index=edge_index, node_labels=node_labels, subgraph_label=label)
        subgraph_list.append(pyg_subgraph)
    return subgraph_list, max_num_node_labels


def subgraph_extraction_labeling(link, graph, hop):
    nodes = set(list([link[0], link[1]]))
    visited = set(list([link[0], link[1]]))
    fringe = set(list([link[0], link[1]]))
    nodes_dist = [0, 0]
    for dist in range(1, hop + 1):
        fringe = neighbors(fringe, graph)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    nodes.remove(link[0])
    nodes.remove(link[1])
    nodes = [link[0], link[1]] + list(nodes)
    subgraph = graph[nodes, :][:, nodes]
    node_labels = DRNL(subgraph)
    g = nx.from_scipy_sparse_array(subgraph)
    if not g.has_edge(0, 1):
        g.add_edge(0, 1)
    return g, node_labels


def neighbors(fringe, A):
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def DRNL(subgraph):
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0] + list(range(2, K)), :][:, [0] + list(range(2, K))]
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels > 1e6] = 0
    labels[labels < -1e6] = 0
    return labels


def to_hypergraphs(graphs, max_num_node_labels, batch, shuffle):
    if shuffle:
        random.shuffle(graphs)
    batch_hypergraphs = []
    batch_node_feats = []
    batch_edge_feats = []
    batch_edges = []
    batch_labels = []
    batch_marks = []
    batch_node_marks = []
    b = 1
    i = 0
    mark = 0
    bias = 0
    for graph in tqdm(graphs, ncols=60):
        i += 1
        node_feats = labels_to_feats(graph.node_labels, max_num_node_labels)
        edge_feats = get_edge_feats(node_feats, graph.edge_index)
        batch_node_feats.append(node_feats)
        batch_edge_feats.append(edge_feats)
        batch_edges.append(graph.edge_index + bias)
        batch_labels.append(graph.subgraph_label)
        batch_marks.append(mark)
        batch_node_marks.append(bias)
        if b == batch or i == len(graphs):
            tensor_batch_node_feats = torch.cat(batch_node_feats, dim=0)
            tensor_batch_edge_feats = torch.cat(batch_edge_feats, dim=0)
            tensor_batch_edges = torch.cat(batch_edges, dim=1)
            tensor_batch_labels = torch.tensor(batch_labels, dtype=torch.long)
            tensor_batch_marks = torch.tensor(batch_marks, dtype=torch.long)
            tensor_batch_node_marks = torch.tensor(batch_node_marks, dtype=torch.long)
            tensor_batch_hyperedges = DHT(tensor_batch_edges)
            batch_hypergraph = Data(x=tensor_batch_edge_feats,
                                    edge_index=tensor_batch_hyperedges,
                                    labels=tensor_batch_labels,
                                    marks=tensor_batch_marks,
                                    edge_x=tensor_batch_node_feats,
                                    edge_marks=tensor_batch_node_marks
                                    )
            batch_hypergraphs.append(batch_hypergraph)
            batch_node_feats = []
            batch_edge_feats = []
            batch_edges = []
            batch_labels = []
            batch_marks = []
            batch_node_marks = []
            b = 1
            mark = 0
            bias = 0
        else:
            b += 1
            mark += graph.edge_index.shape[1]
            bias += graph.node_labels.shape[0]
    return batch_hypergraphs


def labels_to_feats(labels, max_labels):
    feats = torch.zeros(labels.shape[0], max_labels + 1)
    feats.scatter_(1, labels, 1).to(torch.float)
    return feats


def get_edge_feats(node_feats, edges):
    src = edges[0]
    tar = edges[1]
    src_feats = node_feats[src]
    tar_feats = node_feats[tar]
    edge_feats = torch.cat([torch.min(src_feats, tar_feats), torch.max(src_feats, tar_feats)], dim=1)
    return edge_feats


def DHT(edge_index):
    num_edge = edge_index.size(1)
    device = edge_index.device
    edge_to_node_index = torch.arange(0, num_edge, 1, device=device).repeat_interleave(2).view(1, -1)
    hyperedge_index = edge_index.T.reshape(1, -1)
    hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long()
    return hyperedge_index
