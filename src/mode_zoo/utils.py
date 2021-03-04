import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_sparse import SparseTensor

class EdgeDropout(nn.Module):
    def __init__(self, keep_prob=0.5):
        super(EdgeDropout, self).__init__()
        assert keep_prob>0
        self.keep_prob = keep_prob
        self.register_buffer("p", torch.tensor(keep_prob))

    def forward(self, edge_index, edge_weight):
        if self.training:
            mask = torch.rand(edge_index.shape[1])
            mask = torch.floor(mask+self.p).type(torch.bool)
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]/self.p
        return edge_index, edge_weight

    def forward2(self, edge_index):
        if self.training:
            mask = ((torch.rand(edge_index._values().size()) + (self.keep_prob)).floor()).type(torch.bool)
            rc = edge_index._indices()[:, mask]
            val = edge_index._values()[mask]/self.p
            return torch.sparse.FloatTensor(rc, val)
        return edge_index

    def __repr__(self):
        return '{}(keep_prob={})'.format(self.__class__.__name__, self.keep_prob)


class GCN(nn.Module):
    def __init__(self, in_dim=256, h_dims=(128, 64),n_layer=3, add_self_loops=False, use_sparse=True, **kwargs):
        super(GCN, self).__init__()
        self.gcn_layer = nn.ModuleList()
        self.use_sparse = use_sparse
        # self.bns = nn.ModuleList()
        self.adj_dropout = EdgeDropout(keep_prob=0.9)
        self.bn_layer = nn.ModuleList()
        self.gcn_layer.append(gnn.GCNConv(in_dim, h_dims[0],
                                          add_self_loops=add_self_loops,
                                          cached=False))
        # nn.init.kaiming_normal_(self.gcn_layer[0].weight)
        self.bn_layer.append(nn.BatchNorm1d(h_dims[0]))
        for i in range(n_layer-1):
            self.gcn_layer.append(gnn.GCNConv(h_dims[0], h_dims[0],
                                              add_self_loops=add_self_loops,
                                              cached=False))
            self.bn_layer.append(nn.BatchNorm1d(h_dims[0]))
            # self.bns.append(nn.BatchNorm1d(in_dim))
        models = [nn.BatchNorm1d(h_dims[0])]
        for dim_in, dim_out in zip(list(h_dims), h_dims[1:]):
            models.append(nn.Linear(dim_in, dim_out))
            # models.append(nn.BatchNorm1d(dim_out))
            models.append(nn.ReLU(inplace=True))
            # models.append(nn.BatchNorm1d(dim_out))

        # models[-1] = nn.LeakyReLU(inplace=True)
        # del models[-1]
        self.project = nn.Sequential(*models)
        self.bn = nn.BatchNorm1d(in_dim)
        # self.bns = nn.ModuleList()


    def forward(self, x, edge_index, edge_weight=None):
        x = self.bn(x)
        # edge_index, edge_weight = self.adj_dropout(edge_index=edge_index, edge_weight=edge_weight)
        if self.use_sparse:
            edge_index = SparseTensor(row=edge_index[0],
                                  col=edge_index[1],
                                  value=edge_weight,
                                  sparse_sizes=(x.shape[0], x.shape[0]))
            # x = edge_index.t().to_dense()@x
            for i, layer in enumerate(self.gcn_layer):
                x = layer(x=x, edge_index=edge_index.t(), edge_weight=edge_weight)
                # x = self.bn_layer[i](x)
                x = F.relu(x, inplace=True)
                # x = layer(x=x, edge_index=edge_index.t(), edge_weight=edge_weight)
                # x = F.relu(x, inplace=True)
                # x = self.bns[i](x)
                # x = F.dropout(x, 0.1)
        else:
            for layer in self.gcn_layer:
                x = layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
                x = F.relu(x, inplace=True)
        # x = self.bn_layer[0](x)
        x = self.project(x)
        return x

