import torch
from torch import nn, optim
from mode_zoo import bgnn as gnn
# from torch_geometric import nn as gnn
# from torch_sparse import SparseTensor
from mode_zoo.model import SubModel
from mode_zoo.utils import EdgeDropout

def SparseTensor(row, col, value, sparse_sizes):
    return torch.sparse_coo_tensor(indices=torch.stack([row, col]),
                                   values=value,
                                   size=sparse_sizes)

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

class SingleGCN(nn.Module):
    def __init__(self, size_u, size_v, in_channels=64, out_channels=64, normalize=True,
                 dropout=0.4, use_sparse=True, act=nn.ReLU, cached=False, bias=False,
                 **kwargs):
        super(SingleGCN, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.num_nodes = size_u + size_v
        self.use_sparse = use_sparse
        self.dropout = nn.Dropout(dropout)
        self.encoder = gnn.GCNConv(in_channels=in_channels, out_channels=out_channels,
                                   normalize=normalize, add_self_loops=False,
                                   cached=cached, bias=bias, **kwargs)
        self.act = act(inplace=True) if act else nn.Identity()

    def forward(self, x, edge_index, edge_weight):
        x = self.dropout(x)
        if self.use_sparse:
            node_nums = self.num_nodes
            edge_index = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=edge_weight,
                                      sparse_sizes=(node_nums, node_nums)).t()
        feature = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        output = self.act(feature)
        return output

class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""

    def __init__(self, size_u, size_v, input_dim=None, dropout=0.4, act=nn.Sigmoid, ):
        super(InnerProductDecoder, self).__init__()
        self.size_u = size_u
        self.size_v = size_v
        self.dropout = nn.Dropout(dropout)
        if input_dim:
            self.weights = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.xavier_uniform_(self.weights.weight)
        self.act = act() if act is not None else nn.Identity()

    def __call__(self, feature):
        feature = self.dropout(feature)
        R = feature[:self.size_u]
        D = feature[self.size_u:]
        if hasattr(self, "weights"):
            D = self.weights(D)
        x = R@D.T
        outputs = self.act(x)
        return outputs, R, D


class DRHGCN(SubModel):
    def __init__(self, size_u, size_v, in_dim=None, input_type="uv",
                 act=nn.ReLU, dropout=0.4, normalize=True, bias=True,
                 embedding_dim=64, edge_dropout=0.2, lr=0.05, layer_num=2,
                 gnn_mode="gcna", **kwargs):
        super(DRHGCN, self).__init__(size_u=size_u, size_v=size_v, in_dim=in_dim,
                                     input_type=input_type, sigmoid_out=True, **kwargs)

        cached = True if edge_dropout==0.0 else False
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.layer_num = layer_num
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
                                           dropout=dropout, act=None)
        self.save_hyperparameters()

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
        score = torch.sigmoid(score)
        return score, u, v

    def loss_fn(self, batch, score, u, v, reduction="mean"):
        label = batch.uv_train_adj
        bce_loss = self.bce_loss_fn(predict=score, label=label, reduction=reduction)
        loss = bce_loss
        loss_info= {"loss_total":loss}
        attention = torch.softmax(self.attention, dim=0)
        for i in range(len(attention)):
            loss_info[f"layer_att_{i}"] = attention[i].mean()
        return loss, loss_info

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=.1 * self.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * self.lr, max_lr=self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]


class IntraGCN(SubModel):
    def __init__(self, size_u, size_v, in_dim=None, input_type="uv",
                 act=nn.ReLU, dropout=0.4, normalize=True, bias=True,
                 embedding_dim=64, edge_dropout=0.2, lr=0.05, layer_num=2,
                 gnn_mode="gcna",
                 **kwargs):
        super(IntraGCN, self).__init__(size_u=size_u, size_v=size_v, in_dim=in_dim,
                                     input_type=input_type, **kwargs)

        cached = True if edge_dropout==0 else False
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.edge_dropout = EdgeDropout(keep_prob=1-edge_dropout)
        self.short_embedding = nn.Linear(in_features=self.in_dim, out_features=self.embedding_dim)

        intra_encoder = [ShareGCN(size_u=size_u, size_v=size_v, in_channels=self.in_dim,
                                    out_channels=embedding_dim, share=False,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode=gnn_mode) ]
        for layer in range(1, layer_num):
            intra_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, in_channels=embedding_dim,
                                    out_channels=embedding_dim, share=False,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode=gnn_mode) )

        self.intra_encoders = nn.ModuleList(intra_encoder)
        self.attention = nn.Parameter(torch.ones(layer_num, 1, 1) / layer_num)
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0,
                                           dropout=dropout, act=None)
        self.save_hyperparameters()

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight,
                ur_edge_index, ur_edge_weight, vr_edge_index, vr_edge_weight):

        short_embedding = self.short_embedding(x)
        layer_out = [short_embedding]
        for intra_encoder in self.intra_encoders:
            intra_feature = intra_encoder(x, u_edge_index=u_edge_index, u_edge_weight=u_edge_weight,
                                            v_edge_index=v_edge_index, v_edge_weight=v_edge_weight)

            x = intra_feature+layer_out[-1]
            layer_out.append(x)
        x = torch.stack(layer_out[1:])
        attention = torch.softmax(self.attention, dim=0)
        x = torch.sum(x * attention, dim=0)
        score, u, v = self.decoder(x)
        score = torch.sigmoid(score)
        return score, u, v

    def loss_fn(self, batch, score, u, v, reduction="mean"):
        label = batch.uv_train_adj
        bce_loss = self.bce_loss_fn(predict=score, label=label, reduction=reduction)
        loss = bce_loss
        loss_info= {"loss_total":loss}
        attention = torch.softmax(self.attention, dim=0)
        for i in range(len(attention)):
            loss_info[f"layer_att_{i}"] = attention[i].mean()
        return loss, loss_info

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=.1 * self.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * self.lr, max_lr=self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]


class InterGCN(SubModel):
    def __init__(self, size_u, size_v, in_dim=None, input_type="uv",
                 act=nn.ReLU, dropout=0.4, normalize=True, bias=True,
                 embedding_dim=64, edge_dropout=0.2, lr=0.05, layer_num=2,
                 gnn_mode="gcna",
                 **kwargs):
        super(InterGCN, self).__init__(size_u=size_u, size_v=size_v, in_dim=in_dim,
                                     input_type=input_type, **kwargs)

        cached = True if edge_dropout==0 else False
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.edge_dropout = EdgeDropout(keep_prob=1-edge_dropout)
        self.short_embedding = nn.Linear(in_features=self.in_dim, out_features=self.embedding_dim)

        inter_encoder = [ShareGCN(size_u=size_u, size_v=size_v, in_channels=self.in_dim,
                                    out_channels=embedding_dim,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode=gnn_mode)]
        for layer in range(1, layer_num):
            inter_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, in_channels=embedding_dim,
                                    out_channels=embedding_dim,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode=gnn_mode) )
        self.inter_encoders = nn.ModuleList(inter_encoder)
        self.attention = nn.Parameter(torch.ones(layer_num, 1, 1) / layer_num)
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0,
                                           dropout=dropout, act=None)
        self.save_hyperparameters()

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight,
                ur_edge_index, ur_edge_weight, vr_edge_index, vr_edge_weight):

        ur_edge_index, ur_edge_weight = self.edge_dropout(ur_edge_index, ur_edge_weight)
        vr_edge_index, vr_edge_weight = self.edge_dropout(vr_edge_index, vr_edge_weight)

        short_embedding = self.short_embedding(x)
        layer_out = [short_embedding]
        for inter_encoder in self.inter_encoders:
            inter_feature = inter_encoder(x, u_edge_index=ur_edge_index, u_edge_weight=ur_edge_weight,
                                          v_edge_index=vr_edge_index, v_edge_weight=vr_edge_weight)
            x = inter_feature + layer_out[-1]
            layer_out.append(x)
        x = torch.stack(layer_out[1:])
        attention = torch.softmax(self.attention, dim=0)
        x = torch.sum(x * attention, dim=0)
        score, u, v = self.decoder(x)
        score = torch.sigmoid(score)
        return score, u, v

    def loss_fn(self, batch, score, u, v, reduction="mean"):
        label = batch.uv_train_adj
        bce_loss = self.bce_loss_fn(predict=score, label=label, reduction=reduction)
        loss = bce_loss
        loss_info= {"loss_total":loss}
        attention = torch.softmax(self.attention, dim=0)
        for i in range(len(attention)):
            loss_info[f"layer_att_{i}"] = attention[i].mean()
        return loss, loss_info

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=.1 * self.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * self.lr, max_lr=self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]


class MixGCN(SubModel):
    def __init__(self, size_u, size_v, in_dim=None, input_type="uv",
                 act=nn.ReLU, dropout=0.4, normalize=True, bias=True,
                 embedding_dim=64, edge_dropout=0.2, lr=0.05, layer_num=2,
                 gnn_mode="gcna",
                 **kwargs):
        super(MixGCN, self).__init__(size_u=size_u, size_v=size_v, in_dim=in_dim,
                                     input_type=input_type, **kwargs)

        cached = True if edge_dropout==0 else False
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.edge_dropout = EdgeDropout(keep_prob=1-edge_dropout)
        self.short_embedding = nn.Linear(in_features=self.in_dim, out_features=self.embedding_dim)

        encoder = [SingleGCN(size_u=size_u, size_v=size_v, in_channels=self.in_dim,
                            out_channels=embedding_dim,
                            dropout=dropout, act=act, bias=bias,
                            normalize=normalize, cached=cached, gnn_mode=gnn_mode)]
        for layer in range(1, layer_num):
            encoder.append(SingleGCN(size_u=size_u, size_v=size_v, in_channels=embedding_dim,
                                   out_channels=embedding_dim,
                                   dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode=gnn_mode))
        self.encoders = nn.ModuleList(encoder)
        self.attention = nn.Parameter(torch.ones(layer_num, 1, 1) / layer_num)
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0,
                                           dropout=dropout, act=None)
        self.save_hyperparameters()

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight,
                ur_edge_index, ur_edge_weight, vr_edge_index, vr_edge_weight):

        ur_edge_index, ur_edge_weight = self.edge_dropout(ur_edge_index, ur_edge_weight)
        vr_edge_index, vr_edge_weight = self.edge_dropout(vr_edge_index, vr_edge_weight)
        full_edge_index = torch.cat([u_edge_index, ur_edge_index,
                                     vr_edge_index, v_edge_index], dim=1)
        full_edge_weight = torch.cat([u_edge_weight, ur_edge_weight,
                                      vr_edge_weight, v_edge_weight])

        short_embedding = self.short_embedding(x)
        layer_out = [short_embedding]
        for encoder in self.encoders:
            feature = encoder(x, full_edge_index, full_edge_weight)
            x = feature+layer_out[-1]
            layer_out.append(x)
        x = torch.stack(layer_out[1:])
        attention = torch.softmax(self.attention, dim=0)
        x = torch.sum(x * attention, dim=0)
        score, u, v = self.decoder(x)
        score = torch.sigmoid(score)
        return score, u, v

    def loss_fn(self, batch, score, u, v, reduction="mean"):
        label = batch.uv_train_adj
        bce_loss = self.bce_loss_fn(predict=score, label=label, reduction=reduction)
        loss = bce_loss
        loss_info= {"loss_total":loss}
        attention = torch.softmax(self.attention, dim=0)
        for i in range(len(attention)):
            loss_info[f"layer_att_{i}"] = attention[i].mean()
        return loss, loss_info

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=.1 * self.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * self.lr, max_lr=self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]

class StackGCN(SubModel):
    def __init__(self, size_u, size_v, in_dim=None, input_type="uv",
                 act=nn.ReLU, dropout=0.4, normalize=True, bias=True,
                 embedding_dim=64, edge_dropout=0.2, lr=0.05, layer_num=2,
                 gnn_mode="gcna", **kwargs):
        super(StackGCN, self).__init__(size_u=size_u, size_v=size_v, in_dim=in_dim,
                                     input_type=input_type, **kwargs)

        cached = True if edge_dropout==0.0 else False
        self.lr = lr
        self.embedding_dim = embedding_dim
        self.edge_dropout = EdgeDropout(keep_prob=1-edge_dropout)
        self.short_embedding = nn.Linear(in_features=self.in_dim, out_features=self.embedding_dim)

        intra_encoder = [ShareGCN(size_u=size_u, size_v=size_v, in_channels=self.in_dim,
                                    out_channels=embedding_dim, share=False,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode=gnn_mode) ]
        inter_encoder = [ShareGCN(size_u=size_u, size_v=size_v, in_channels=embedding_dim,
                                    out_channels=embedding_dim,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode=gnn_mode)]
        for layer in range(1, layer_num):
            intra_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, in_channels=embedding_dim,
                                    out_channels=embedding_dim, share=False,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached, gnn_mode=gnn_mode) )

            inter_encoder.append(ShareGCN(size_u=size_u, size_v=size_v, in_channels=embedding_dim,
                                    out_channels=embedding_dim,
                                    dropout=dropout, act=act, bias=bias,
                                    normalize=normalize, cached=cached,gnn_mode=gnn_mode) )
        self.intra_encoders = nn.ModuleList(intra_encoder)
        self.inter_encoders = nn.ModuleList(inter_encoder)
        self.attention = nn.Parameter(torch.ones(layer_num, 1, 1) / layer_num)
        self.decoder = InnerProductDecoder(size_u=size_u, size_v=size_v, input_dim=0,
                                           dropout=dropout, act=None)
        self.save_hyperparameters()

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight,
                ur_edge_index, ur_edge_weight, vr_edge_index, vr_edge_weight):

        ur_edge_index, ur_edge_weight = self.edge_dropout(ur_edge_index, ur_edge_weight)
        vr_edge_index, vr_edge_weight = self.edge_dropout(vr_edge_index, vr_edge_weight)

        short_embedding = self.short_embedding(x)
        layer_out = [short_embedding]
        for inter_encoder, intra_encoder in zip(self.inter_encoders, self.intra_encoders):
            intra_feature = intra_encoder(x, u_edge_index=u_edge_index, u_edge_weight=u_edge_weight,
                                          v_edge_index=v_edge_index, v_edge_weight=v_edge_weight)
            inter_feature = inter_encoder(intra_feature, u_edge_index=ur_edge_index, u_edge_weight=ur_edge_weight,
                                          v_edge_index=vr_edge_index, v_edge_weight=vr_edge_weight)
            x = inter_feature + layer_out[-1]
            layer_out.append(x)
        x = torch.stack(layer_out[1:])
        attention = torch.softmax(self.attention, dim=0)
        x = torch.sum(x * attention, dim=0)
        score, u, v = self.decoder(x)
        score = torch.sigmoid(score)
        return score, u, v
