import torch
import warnings
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics
from model import HGLP
from tqdm import tqdm
from utils import fix_seed, load_graph, sample_neg, get_train_graph, links_to_subgraphs, to_hypergraphs

warnings.filterwarnings('ignore')
torch.set_printoptions(profile="full")
device = torch.device('cuda:0')

if __name__ == "__main__":
    f = open("./log.txt", 'w')
    for graph_name in ['Celegans', 'USAir', 'SMG', 'EML', 'YST', 'Power', 'GRQ']:
        mean_best_auc = []
        mean_best_ap = []
        for seed in range(1, 2):
            test_ratio = 0.2
            hop = 2
            batch = 50
            latent_dim = [32, 32, 32, 1]
            hidden_size = 128
            dropout = True
            num_epochs = 15
            learning_rate = 0.005

            fix_seed(seed)

            graph = load_graph(graph_name)

            train_pos, train_neg, test_pos, test_neg = sample_neg(graph, 0.2)

            train_graph = get_train_graph(graph, test_pos)

            train_subgraphs, test_subgraphs, max_num_node_labels = links_to_subgraphs(train_graph, train_pos, train_neg, test_pos, test_neg, hop)

            train_hypergraphs = to_hypergraphs(train_subgraphs, max_num_node_labels, batch, True)
            test_hypergraphs = to_hypergraphs(test_subgraphs, max_num_node_labels, batch, False)

            model = HGLP(int(max_num_node_labels + 1), hidden_size, latent_dim, dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            best_auc = 0
            best_ap = 0
            for epoch in range(num_epochs):
                total_loss = []
                all_targets = []
                all_scores = []

                model.train()
                for g in tqdm(train_hypergraphs, ncols=100):
                    g.to(device)
                    x, edge_index, y, marks, edge_x, edge_marks = g.x, g.edge_index, g.labels, g.marks, g.edge_x, g.edge_marks
                    out = model(x, edge_index, marks, edge_x, edge_marks)
                    loss = F.nll_loss(out, y)
                    total_loss.append(loss.data.cpu().detach())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                for g in tqdm(test_hypergraphs, ncols=100):
                    g.to(device)
                    x, edge_index, y, marks, edge_x, edge_marks = g.x, g.edge_index, g.labels, g.marks, g.edge_x, g.edge_marks
                    out = model(x, edge_index, marks, edge_x, edge_marks)
                    all_targets.extend(y.tolist())
                    all_scores.append(out[:, 1].cpu().detach())

                total_loss = np.array(total_loss)
                all_targets = np.array(all_targets)
                all_scores = torch.cat(all_scores).cpu().numpy()
                ap = metrics.average_precision_score(all_targets, all_scores)
                fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                print(('average test of epoch %d: loss %.5f auc %.5f ap %.5f' % (
                    epoch, float(np.mean(total_loss)), auc, ap)))
                if auc > best_auc:
                    best_auc = auc
                    best_ap = ap
            print(best_auc, best_ap)
            mean_best_auc.append(best_auc)
            mean_best_ap.append(best_ap)
        print(mean_best_auc, file=f)
        print(sum(mean_best_auc) / 10, file=f)
        print(mean_best_ap, file=f)
        print(sum(mean_best_ap) / 10, file=f)

        print(mean_best_auc)
        print(sum(mean_best_auc) / 10)
        print(mean_best_ap)
        print(sum(mean_best_ap) / 10)
