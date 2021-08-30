import torch
from collections import namedtuple
from .. import DATA_TYPE_REGISTRY
from ..dataloader import Dataset

def gcn_norm(edge_index, add_self_loops=True):
    adj_t = edge_index.to_dense()
    if add_self_loops:
        adj_t = adj_t+torch.eye(*adj_t.shape)
    deg = adj_t.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.)

    adj_t.mul_(deg_inv_sqrt.view(-1, 1))
    adj_t.mul_(deg_inv_sqrt.view(1, -1))
    edge_index = adj_t.to_sparse()
    return edge_index

FullGraphData = namedtuple("FullGraphData", ["u_edge", "v_edge",
                                             "embedding", "edge",
                                             "uv_edge", "vu_edge",
                                              "label", "interaction_pair", "valid_mask"])

@DATA_TYPE_REGISTRY.register()
class FullGraphDataset(Dataset):
    def __init__(self, dataset, mask, fill_unkown=True, **kwargs):
        super(FullGraphDataset, self).__init__(dataset, mask, fill_unkown=True, **kwargs)
        assert fill_unkown, "fill_unkown need True!"
        self.data = self.build_data()

    def build_data(self):
        u_edge = self.get_u_edge(union_graph=True)
        v_edge = self.get_v_edge(union_graph=True)
        uv_edge = self.get_uv_edge(union_graph=True)
        vu_edge = self.get_vu_edge(union_graph=True)
        edge = self.get_union_edge(union_type="u-uv-vu-v")
        x = self.get_union_edge(union_type="u-v")

        x = torch.sparse_coo_tensor(indices=x[0], values=x[1], size=x[2])
        # x = x.to_dense()

        norm_x = gcn_norm(edge_index=x, add_self_loops=False).to_dense()
        x = norm_x * torch.norm(x) / torch.norm(norm_x)

        data = FullGraphData(u_edge=u_edge,
                             v_edge=v_edge,
                             uv_edge=uv_edge,
                             vu_edge=vu_edge,
                             edge=edge,
                             label=self.label,
                             valid_mask=self.valid_mask,
                             interaction_pair=self.interaction_edge,
                             embedding=x,
                             )
        return data

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.data