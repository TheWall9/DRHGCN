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

class GAT(nn.Module):
    def __init__(self, in_dim=256, h_dims=(128,), n_layer=1, num_heads=1, add_self_loops=False, **kwargs):
        assert in_dim%num_heads==0
        super(GAT, self).__init__()
        self.gcn_layer = nn.ModuleList()
        for i in range(n_layer):
            self.gcn_layer.append(gnn.GATConv(in_dim, in_dim//num_heads,
                                              heads=num_heads,
                                              add_self_loops=add_self_loops))
        models = []
        for dim_in, dim_out in zip([in_dim]+list(h_dims), h_dims):
            models.append(nn.Linear(dim_in, dim_out))
            models.append(nn.ReLU(inplace=True))
        self.project = nn.Sequential(*models)

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.gcn_layer:
            x = layer(x=x, edge_index=edge_index)
            x = F.relu(x, inplace=True)
        x = self.project(x)
        return x


class MultiGCN(nn.Module):
    def __init__(self, in_dim=256, h_dims=(128, 64), att_dim=64, n_layer=3, add_self_loops=False, use_sparse=True, **kwargs):
        super(MultiGCN, self).__init__()
        self.gcn_layer = nn.ModuleList()
        self.use_sparse = use_sparse
        self.bn_layer = nn.ModuleList()
        self.att_layer = nn.ModuleList()
        for i in range(n_layer):
            self.gcn_layer.append(gnn.GCNConv(in_dim, in_dim,
                                              add_self_loops=add_self_loops,
                                              cached=True))
            self.att_layer.append(HanAtt(in_dim, att_dim, n_hop=kwargs["n_hop"]))
            self.bn_layer.append(nn.BatchNorm1d(in_dim))
            # self.bns.append(nn.BatchNorm1d(in_dim))
        models = []
        for dim_in, dim_out in zip([in_dim]+list(h_dims), h_dims):
            models.append(nn.Linear(dim_in, dim_out))
            models.append(nn.LeakyReLU(inplace=True))
        self.project = nn.Sequential(*models)
        self.bn = nn.BatchNorm1d(in_dim)
        # self.bns = nn.ModuleList()


    def forward(self, x, edges_index, edges_weight=None):
        x = self.bn(x)
        edges_index = [SparseTensor(row=index[0], col=index[1],
                              value=weight,
                              sparse_sizes=(x.shape[0],x.shape[0]))
                       for index, weight in zip(edges_index, edges_weight)]

        for i, layer in enumerate(self.gcn_layer):
            xs = [layer(x=x, edge_index=index.t()) for index in edges_index]
            # x = self.bn_layer[i](x)
            x = torch.stack(xs)
            x = self.att_layer[i](x)
            x = F.relu(x, inplace=True)
            # x = self.bns[i](x)
            # x = F.dropout(x, 0.1)
        x = self.project(x)
        return x


class HanAtt(nn.Module):
    def __init__(self, in_dim, h_dim, n_hop=1):
        super(HanAtt, self).__init__()
        self.h_dim = h_dim
        self.linears = nn.ModuleList()
        for _ in range(n_hop):
            self.linears.append(nn.Linear(in_dim, h_dim, bias=True))
        self.qs = nn.ModuleList()
        for _ in range(n_hop):
            self.qs.append(nn.Linear(h_dim, 1, bias=False))

    def forward(self, vals):
        """
        :param val: [K, N, D]
        :return:
        """
        sims = []
        for q, linear, val in zip(self.qs, self.linears, vals):
            key = torch.tanh(linear(val))
            sim = q(key)
            sims.append(sim)
        sim = torch.stack(sims)
        alpha = torch.softmax(sim, dim=0).squeeze(-1)
        # alpha:[k, N]
        output = torch.einsum("knd,kn->nd", vals, alpha)
        return output


class MutilSelfGCN(nn.Module):
    def __init__(self, in_dim=256, h_dims=(128, 64), num_heads=8, n_layer=3, add_self_loops=False, use_sparse=True, **kwargs):
        super(MutilSelfGCN, self).__init__()
        self.gcn_layer = nn.ModuleList()
        self.use_sparse = use_sparse
        self.bn_layer = nn.ModuleList()
        self.att_layer = nn.ModuleList()
        for i in range(n_layer):
            self.gcn_layer.append(gnn.GCNConv(in_dim, in_dim,
                                              add_self_loops=add_self_loops,
                                              cached=True))
            self.att_layer.append(nn.MultiheadAttention(in_dim, num_heads=num_heads))
            self.bn_layer.append(nn.BatchNorm1d(in_dim))
            # self.bns.append(nn.BatchNorm1d(in_dim))
        models = []
        for dim_in, dim_out in zip([in_dim]+list(h_dims), h_dims):
            models.append(nn.Linear(dim_in, dim_out))
            models.append(nn.LeakyReLU(inplace=True))
        self.project = nn.Sequential(*models)
        self.bn = nn.BatchNorm1d(in_dim)
        # self.bns = nn.ModuleList()


    def forward(self, x, edges_index, edges_weight=None):
        x = self.bn(x)
        edges_index = [SparseTensor(row=index[0], col=index[1],
                              value=weight,
                              sparse_sizes=(x.shape[0],x.shape[0]))
                       for index, weight in zip(edges_index, edges_weight)]

        for i, layer in enumerate(self.gcn_layer):
            xs = [layer(x=x, edge_index=index.t()) for index in edges_index]
            # x = self.bn_layer[i](x)
            x = torch.stack(xs)
            x = self.att_layer[i](query=x, key=x, value=x)
            x = F.relu(x, inplace=True)
            # x = self.bns[i](x)
            # x = F.dropout(x, 0.1)

        x = self.project(x)
        return x

class LuongAtt(nn.Module):
    def __init__(self, h_dim, att_type=""):
        assert att_type in ("dot", "general", "concat")
        super(LuongAtt, self).__init__()
        self.att_type = att_type
        if att_type=="general":
            self.w = nn.Linear(h_dim, h_dim, bias=False)
        elif att_type=="concat":
            self.w = nn.Linear(2*h_dim, h_dim, bias=False)
            self.v = nn.Linear(h_dim, 1, bias=False)

    def forward(self, query, key):
        """
        :param query: [N, D]
        :param key: [K, N, D]
        :return:
        """
        query = query.unsqueeze(0).repeat(key.shape[0], 1, 1)
        # score:[K,N]
        if self.att_type=="dot":
            score = torch.sum(query*key, dim=-1)
        elif self.att_type=="general":
            score = torch.sum(query*self.w(key), dim=-1)
        elif self.att_type=="concat":
            feature = torch.cat([query, key], dim=-1)
            score = self.v(torch.tanh(self.w(feature))).squeeze(-1)
        score = torch.softmax(score, dim=0)
        # print(score.mean(dim=1))
        output = torch.einsum("knd,kn->nd", key, score)
        return output

class LuongGCN(nn.Module):
    def __init__(self, in_dim=256, h_dims=(128, 64), n_layer=3, add_self_loops=False, use_sparse=True, **kwargs):
        super(LuongGCN, self).__init__()
        self.gcn_layer = nn.ModuleList()
        self.use_sparse = use_sparse
        self.bn_layer = nn.ModuleList()
        self.att_layer = nn.ModuleList()
        for i in range(n_layer):
            self.gcn_layer.append(gnn.GCNConv(in_dim, in_dim,
                                              add_self_loops=add_self_loops,
                                              cached=True))
            self.att_layer.append(LuongAtt(in_dim, att_type=kwargs.get("att_type", "dot")))
            self.bn_layer.append(nn.BatchNorm1d(in_dim))
            # self.bns.append(nn.BatchNorm1d(in_dim))
        models = []
        for dim_in, dim_out in zip([in_dim]+list(h_dims), h_dims):
            models.append(nn.Linear(dim_in, dim_out))
            models.append(nn.LeakyReLU(inplace=True))
        self.project = nn.Sequential(*models)
        self.bn = nn.BatchNorm1d(in_dim)
        # self.bns = nn.ModuleList()


    def forward(self, x, edges_index, edges_weight=None):
        x = self.bn(x)
        edges_index = [SparseTensor(row=index[0], col=index[1],
                              value=weight,
                              sparse_sizes=(x.shape[0],x.shape[0]))
                       for index, weight in zip(edges_index, edges_weight)]

        for i, layer in enumerate(self.gcn_layer):
            xs = [layer(x=x, edge_index=index.t()) for index in edges_index]
            # x = self.bn_layer[i](x)
            xs = torch.stack(xs)
            x = self.att_layer[i](query=x, key=xs)
            x = F.relu(x, inplace=True)
            # x = self.bns[i](x)
            # x = F.dropout(x, 0.1)

        x = self.project(x)
        return x

