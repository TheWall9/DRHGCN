
"""https://github.com/zhuhm1996/bgnn"""

import torch
from torch import nn, optim
from mode_zoo import gnn
from mode_zoo.utils import EdgeDropout

def bgnn_pool(xw, adj):
    sum = adj@xw
    sum_squared = sum.square()
    # step2 squared_sum
    squared = xw.square()
    squared_sum = torch.square(adj)@squared
    # step3
    new_embedding = sum_squared - squared_sum
    return new_embedding

def bgcn_a_norm(edge_index):
    adj_t = edge_index.to_dense()
    adj_all = adj_t+torch.eye(adj_t.shape[0])
    num_nei = adj_all.sum(dim=-1)
    norm = (adj_all.sum(dim=-1).square()-adj_all.square().sum(dim=-1))
    # norm = num_nei*(num_nei-1)
    norm = norm.pow(-1)
    norm.masked_fill_(torch.isinf(norm), 0.)
    norm = torch.diag(norm)
    norm = norm.to_sparse()
    adj_all = adj_all.to_sparse()
    return adj_all, norm

def bgcn_t_norm(edge_index):
    adj_t = edge_index.to_dense()
    adj_all = adj_t+torch.eye(adj_t.shape[0])
    norm = adj_t.sum(dim=-1)
    norm = norm.pow(-1)
    norm.masked_fill_(torch.isinf(norm), 0.)
    norm = torch.diag(norm)
    norm = norm.to_sparse()
    adj_all = adj_all.to_sparse()
    return adj_all, norm


class BGCNA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):
        super(BGCNA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self._cache = None
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     nn.init.xavier_uniform_(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_weight):
        xw = x@self.weight
        if self.cached:
            if self._cache is None:
                adj, norm = bgcn_a_norm(edge_index)
                self._cache = (adj, norm)
            else:
                adj, norm = self._cache
        else:
            adj, norm = bgcn_a_norm(edge_index)
        out = bgnn_pool(xw, adj)
        out = norm@out
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class BGCNT(BGCNA):
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):
        super(BGCNT, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                    cached=cached, bias=bias, **kwargs)

    def forward(self, x, edge_index, edge_weight):
        xw = x @ self.weight
        if self.cached:
            if self._cache is None:
                adj, norm = bgcn_a_norm(edge_index)
                self._cache = (adj, norm)
            else:
                adj, norm = self._cache
        else:
            adj, norm = bgcn_t_norm(edge_index)
        out = bgnn_pool(xw, adj) - bgnn_pool(xw, edge_index)
        out = norm @ out
        if self.bias is not None:
            out += self.bias
        return xw-out

class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 gnn_mode="gcnt", improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):
        super(GCNConv, self).__init__()
        assert gnn_mode in ["gcn", "gcna", "gcnt", "a", "t"]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = gnn_mode
        if gnn_mode.startswith("gcn"):
            self.gcn = gnn.GCNConv(in_channels=in_channels, out_channels=out_channels,
                                normalize=normalize, add_self_loops=add_self_loops,
                                cached=cached, bias=bias)
            self.weight = self.gcn.weight
        if gnn_mode.endswith("a"):
            self.bgnn = BGCNA(in_channels=in_channels, out_channels=out_channels, cached=cached, bias=bias)
            self.weight = self.bgnn.weight
        else:
            self.bgnn = BGCNT(in_channels=in_channels, out_channels=out_channels, cached=cached, bias=bias)
            self.weight = self.bgnn.weight
        if len(gnn_mode)==4:
            self.attention = nn.Parameter(torch.ones(2, 1, 1)/2)

    def forward(self, x, edge_index, edge_weight):
        if self.mode.startswith("gcn"):
            x_gcn = self.gcn(x, edge_index, edge_weight)
        if self.mode.endswith("a") or self.mode.endswith("t"):
            x_bgnn = self.bgnn(x, edge_index, edge_weight)
        if len(self.mode)==1:
            return x_bgnn
        elif len(self.mode)==3:
            return x_gcn
        attention = torch.softmax(self.attention, dim=0)
        feature = torch.stack([x_gcn, x_bgnn])
        feature = torch.sum(attention*feature, dim=0)
        return feature
