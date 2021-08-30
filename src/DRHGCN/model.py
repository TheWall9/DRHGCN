import torch
from torch import nn, optim
from functools import partial
from ..model_help import BaseModel
from .dataset import FullGraphData
from .. import MODEL_REGISTRY
from . import gnn


def SparseTensor(row, col, value, sparse_sizes):
    return torch.sparse_coo_tensor(indices=torch.stack([row, col]),
                                   values=value,
                                   size=sparse_sizes)

class EdgeDropout(nn.Module):
    def __init__(self, keep_prob=0.5):
        super(EdgeDropout, self).__init__()
        assert keep_prob>0
        self.keep_prob = keep_prob
        self.register_buffer("p", torch.tensor(keep_prob))

    def forward(self, edge_index, edge_weight):
        if self.training:
            mask = torch.rand(edge_index.shape[1], device=edge_weight.device)
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


class ShareGCN(nn.Module):
    def __init__(self, size_u, size_v, in_channels=64, out_channels=64, share=True, normalize=True,
                 dropout=0.4, use_sparse=True, act=nn.ReLU, cached=False, bias=False, add_self_loops=False,
                 **kwargs):
        super(ShareGCN, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.num_nodes = size_u+size_v
        self.share = share
        self.use_sparse = use_sparse
        self.dropout = nn.Dropout(dropout)
        self.u_encoder = gnn.GCNConv(in_channels=in_channels, out_channels=out_channels,
                                     normalize=normalize, add_self_loops=add_self_loops,
                                     cached=cached, bias=bias, **kwargs)
        if not self.share:
            self.v_encoder = gnn.GCNConv(in_channels=in_channels, out_channels=out_channels,
                                         normalize=normalize, add_self_loops=False,
                                         cached=cached, bias=bias, **kwargs)
        self.act = act(inplace=True) if act else nn.Identity()

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight):
        x = self.dropout(x)
        if self.share:
            edge_index = torch.cat([u_edge_index, v_edge_index], dim=1)
            edge_weight = torch.cat([u_edge_weight, v_edge_weight], dim=0)
            if self.use_sparse:
                node_nums = self.num_nodes
                edge_index = SparseTensor(row=edge_index[0], col=edge_index[1],
                                          value=edge_weight,
                                          sparse_sizes=(node_nums, node_nums)).t()
            feature = self.u_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            if self.use_sparse:
                node_nums = self.num_nodes
                u_edge_index = SparseTensor(row=u_edge_index[0], col=u_edge_index[1],
                                           value=u_edge_weight,
                                           sparse_sizes=(node_nums, node_nums)).t()
                v_edge_index = SparseTensor(row=v_edge_index[0], col=v_edge_index[1],
                                            value=v_edge_weight,
                                            sparse_sizes=(node_nums, node_nums)).t()
            feature_u = self.u_encoder(x=x, edge_index=u_edge_index, edge_weight=u_edge_weight)
            feature_v = self.v_encoder(x=x, edge_index=v_edge_index, edge_weight=v_edge_weight)
            feature = torch.cat([feature_u[:self.size_u], feature_v[self.size_u:]])
        output = self.act(feature)
        return output

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, size_u, size_v, input_dim=None, dropout=0.4, act=nn.Sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.dropout = nn.Dropout(dropout)
        if input_dim:
            self.weights = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights.weight)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, feature):
        feature = self.dropout(feature)
        R = feature[:self.size_u]
        D = feature[self.size_u:]
        if hasattr(self, "weights"):
            D = self.weights(D)
        x = R@D.T
        outputs = self.act(x)
        return outputs, R, D


class SmoothDecoder(nn.Module):
    def __init__(self, size_u, size_v, k=20, act=nn.Sigmoid):
        super(SmoothDecoder, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.k = k
        self.act = act() if act is not None else nn.Identity()

    def merge_neighbor_feature(self, sims, features, k=5):
        assert sims.shape[0] == features.shape[0] and sims.shape[1] == sims.shape[0]
        if k<0:
            k = sims.shape[1]
        N = features.shape[0]
        value, idx = torch.topk(sims, dim=1, k=k)
        col = idx.reshape(-1)
        features = features[col].view(N, k, -1) * value.view(N, k, 1)
        features = features.sum(dim=1)
        features = features / value.sum(dim=1).view(N, 1)
        return features

    def forward(self, u, v, batch:FullGraphData):
        if not hasattr(self, "sim"):
            indices = torch.cat([batch.u_edge[0], batch.v_edge[0]], dim=1)
            values = torch.cat([batch.u_edge[1], batch.v_edge[1]], dim=0)
            size = batch.u_edge[2]
            sim = torch.sparse_coo_tensor(indices, values, size).to_dense()
            self.register_buffer("sim", sim)
        if not hasattr(self, "mask"):
            if self.training:
                feature = torch.cat([u, v], dim=0)
                interactions = torch.sparse_coo_tensor(indices=batch.interaction_pair,
                                                       values=batch.label.reshape(-1),
                                                       size=(self.size_u, self.size_v)).to_dense()
                index = torch.nonzero(interactions)
                u_idx, v_idx = index[:, 0].unique(), index[:, 1].unique()
                v_idx = v_idx+self.size_u
                mask = torch.zeros(feature.shape[0], 1, dtype=torch.bool, device=feature.device)
                mask[u_idx] = True
                mask[v_idx] = True
        elif not self.training:
            feature = torch.cat([u, v], dim=0)
            merged_feature = self.merge_neighbor_feature(self.sim, feature, self.k)
            feature = torch.where(self.mask, feature, merged_feature)
            u = feature[:self.size_u]
            v = feature[self.size_u:]
        x = u@v.T
        outputs = self.act(x)
        return outputs, u, v


@MODEL_REGISTRY.register()
class DRHGCN(BaseModel):
    DATASET_TYPE = "FullGraphDataset"

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DRHGCN model config")
        parser.add_argument("--embedding_dim", default=128, type=int)
        parser.add_argument("--layer_num", default=2, type=int)
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--edge_dropout", default=0.2, type=float)
        parser.add_argument("--neighbor_num", type=int, default=15)
        parser.add_argument("--smooth", default=False, action="store_true")
        parser.add_argument("--gnn_mode", default="gcnt")
        return parent_parser

    def __init__(self, size_u, size_v,
                 act=nn.ReLU, dropout=0.4, normalize=True, bias=True,
                 embedding_dim=64, edge_dropout=0.2, lr=0.05, layer_num=2,
                 pos_weight=1.0,
                 gnn_mode="gcnt", smooth=False, **kwargs):
        super(DRHGCN, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.num_nodes = size_u + size_v
        self.use_embedding = False
        self.in_dim = self.num_nodes
        self.smooth = smooth

        cached = True if edge_dropout==0.0 else False
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.layer_num = layer_num

        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.loss_fn = partial(self.bce_loss_fn, pos_weight=self.pos_weight)

        self.edge_dropout = EdgeDropout(keep_prob=1-edge_dropout)
        self.short_embedding = nn.Linear(in_features=self.in_dim, out_features=self.embedding_dim)

        intra_encoder = [ShareGCN(size_u=size_u, size_v=size_v, in_channels=self.in_dim,
                                    out_channels=embedding_dim, share=False,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode="gcn") ]
        inter_encoder = [ShareGCN(size_u=size_u, size_v=size_v, in_channels=self.in_dim,
                                    out_channels=embedding_dim,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode=gnn_mode)]
        for layer in range(1, layer_num):
            intra_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, in_channels=embedding_dim,
                                    out_channels=embedding_dim, share=False,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode="gcn") )

            inter_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, in_channels=embedding_dim,
                                    out_channels=embedding_dim,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached,gnn_mode=gnn_mode) )
        self.intra_encoders = nn.ModuleList(intra_encoder)
        self.inter_encoders = nn.ModuleList(inter_encoder)
        self.attention = nn.Parameter(torch.ones(layer_num, 1, 1) / layer_num)
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0,
                                           dropout=dropout)
        self.smooth_decoder = SmoothDecoder(size_u=size_u, size_v=size_v, k=kwargs["neighbor_num"])
        self.save_hyperparameters()

    def step(self, batch:FullGraphData):
        x = batch.embedding
        u_edge_index, u_edge_weight = batch.u_edge[:2]
        v_edge_index, v_edge_weight = batch.v_edge[:2]
        ur_edge_index, ur_edge_weight = batch.uv_edge[:2]
        vr_edge_index, vr_edge_weight = batch.vu_edge[:2]
        label = batch.label
        predict, u, v = self.forward(x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight,
                ur_edge_index, ur_edge_weight, vr_edge_index, vr_edge_weight)
        if self.smooth:
            predict, u, v = self.smooth_decoder(u, v, batch)

        if not self.training:
            predict = predict[batch.valid_mask.reshape(*predict.shape)]
            label = label[batch.valid_mask]
        ans = self.loss_fn(predict=predict, label=label)
        ans["predict"] = predict.reshape(-1)
        ans["label"] = label.reshape(-1)
        return ans


    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight,
                ur_edge_index, ur_edge_weight, vr_edge_index, vr_edge_weight):
        ur_edge_index, ur_edge_weight = self.edge_dropout(ur_edge_index, ur_edge_weight)
        vr_edge_index, vr_edge_weight = self.edge_dropout(vr_edge_index, vr_edge_weight)

        short_embedding = self.short_embedding(x)
        layer_out = [short_embedding]
        for inter_encoder, intra_encoder in zip(self.inter_encoders, self.intra_encoders):
            intra_feature = intra_encoder(x, u_edge_index=u_edge_index, u_edge_weight=u_edge_weight,
                                          v_edge_index=v_edge_index, v_edge_weight=v_edge_weight)
            inter_feature = inter_encoder(x, u_edge_index=ur_edge_index, u_edge_weight=ur_edge_weight,
                                          v_edge_index=vr_edge_index, v_edge_weight=vr_edge_weight)
            x = intra_feature + inter_feature + layer_out[-1]
            layer_out.append(x)
        x = torch.stack(layer_out[1:])
        attention = torch.softmax(self.attention, dim=0)
        x = torch.sum(x * attention, dim=0)

        score, u, v = self.decoder(x)
        # score = torch.sigmoid(score)
        return score, u, v

    def training_step(self, batch, batch_idx=None):
        return self.step(batch)

    def validation_step(self, batch, batch_idx=None):
        return self.step(batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=.1 * self.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * self.lr, max_lr=self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]
