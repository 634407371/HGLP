import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import HypergraphConv


class HGLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim, with_dropout):
        super(HGLP, self).__init__()
        self.latent_dim = latent_dim

        self.conv_for_node = nn.ModuleList()
        self.conv_for_node.append(HypergraphConv(input_dim * 2, latent_dim[0] * 2))
        for i in range(1, len(latent_dim)):
            self.conv_for_node.append(HypergraphConv(latent_dim[i - 1] * 2, latent_dim[i] * 2))

        self.conv_for_edge = nn.ModuleList()
        self.conv_for_edge.append(HypergraphConv(input_dim, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_for_edge.append(HypergraphConv(latent_dim[i - 1], latent_dim[i]))

        latent_dim = sum(latent_dim) * 4

        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

        self.with_dropout = with_dropout

    def forward(self, x, edge_index, marks, edge_x, edge_marks):
        all_x = []
        all_edge_x = []
        lv = 0
        while lv < len(self.latent_dim):
            x_next = self.conv_for_node[lv](x, edge_index)
            x_next = F.relu(x_next)
            all_x.append(x_next)

            edge_x = self.conv_for_edge[lv](edge_x, edge_index[[1, 0]])
            edge_x = F.relu(edge_x)
            all_edge_x.append(edge_x)

            x = x_next

            lv += 1

        x = torch.cat(all_x, 1)
        x = x[marks]

        edge_x = torch.cat(all_edge_x, 1)
        edge_x = torch.cat(
            [torch.min(edge_x[edge_marks], edge_x[edge_marks + 1]),
             torch.max(edge_x[edge_marks], edge_x[edge_marks + 1])], 1)

        x = torch.cat([edge_x, x], 1)

        x = self.linear1(x)
        x = F.relu(x)
        if self.with_dropout:
            x = F.dropout(x, training=self.training)
        x = self.linear2(x)
        output = F.log_softmax(x, dim=1)

        return output
