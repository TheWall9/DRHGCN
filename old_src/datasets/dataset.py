import torch
from torch_geometric.data import Data

import os
import numpy as np
from sklearn.model_selection import KFold
from functools import lru_cache
import pandas as pd
import scipy.io as scio
from torch_geometric.data import Dataset

class BipartiteGraph(Data):
    def __init__(self, u_x, u_adj, v_x, v_adj,
                 uv_train_one_index,
                 uv_test_index, uv_test_weight,
                 uv_adj=None):
        super(BipartiteGraph, self).__init__()
        self.dim_u = u_adj.shape[0]
        self.dim_v = v_adj.shape[0]

        self.u_x = u_x
        self.v_x = v_x

        self.u_adj = u_adj
        self.v_adj = v_adj
        self.uv_adj = uv_adj

        self.u_edge_index = (torch.nonzero(u_adj, as_tuple=False).T).contiguous()
        self.u_edge_weight = self.u_adj[self.u_edge_index[0], self.u_edge_index[1]]

        self.v_edge_index = (torch.nonzero(v_adj, as_tuple=False).T).contiguous()
        self.v_edge_weight = self.v_adj[self.v_edge_index[0], self.v_edge_index[1]]

        self.uv_train_one_index = uv_train_one_index
        self.uv_train_one_weight = torch.ones(uv_train_one_index.shape[1])
        self.uv_train_adj = torch.sparse_coo_tensor(indices=uv_train_one_index,
                                                    values=torch.ones(uv_train_one_index.shape[1]),
                                                    size=(u_adj.shape[0], v_adj.shape[0])).to_dense()
        self._uv_train_zero_index = torch.stack(torch.where(self.uv_train_adj < 0.5))
        self._uv_train_zero_weight = torch.zeros(self._uv_train_zero_index.shape[1])
        self.uv_test_edge_index = uv_test_index
        self.uv_test_edge_weight = uv_test_weight

        self.uv_train_zero_index = self._uv_train_zero_index
        self.uv_train_zero_weight = self._uv_train_zero_weight[:self._uv_train_zero_index.shape[1]]
        self.uv_train_edge_index = torch.cat([self.uv_train_one_index, self.uv_train_zero_index], dim=1)
        self.uv_train_edge_weight = self.uv_train_adj[self.uv_train_edge_index[0], self.uv_train_edge_index[1]]

        # self.split_u_adj()
        # self.split_v_adj()


    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self

    def __iter__(self):
        yield self

    def split_u_adj(self, k=20):
        u_adj = select_topk(self.u_adj, k=k)
        # self.u_x = u_adj
        self.u_edge_index = (torch.nonzero(u_adj, as_tuple=False).T).contiguous()
        self.u_edge_weight = u_adj[self.u_edge_index[0], self.u_edge_index[1]]
        # self.u_edge_weight.fill_(1.0)

    def split_v_adj(self, k=20):
        v_adj = select_topk(self.v_adj, k=k)
        # self.v_x = v_adj
        self.v_edge_index = (torch.nonzero(v_adj, as_tuple=False).T).contiguous()
        self.v_edge_weight = v_adj[self.v_edge_index[0], self.v_edge_index[1]]
        # self.v_edge_weight.fill_(1.0)

class BipartiteGraphDataset(Dataset):
    def __init__(self, config, data_loader):
        self.dataset_name = config.get("dataset_name")
        self.n_splits = config.get("n_splits")
        self.mode = config.get("mode", "global")
        self.u_idx = config.get("u_idx")
        self.v_idx = config.get("v_idx")
        self.seed = config.get("seed", 666)
        self.shuffle = config.get("shuffle", True)
        self.split_zero = config.get("split_zero", True)
        self.uk = config.get("uk", 20)
        self.vk = config.get("vk", 20)
        self.u_adj, self.v_adj, self.uv_adj, self.u_name, self.v_name = data_loader()
        self.n_splits = min(self.n_splits, self.u_adj.shape[0]) if self.u_idx else min(self.n_splits, self.v_adj.shape[0])
        assert self.u_adj.shape[0]==self.uv_adj.shape[0]
        assert self.v_adj.shape[1]==self.uv_adj.shape[1]
        assert self.u_adj.shape[0]==self.u_adj.shape[1]
        assert self.v_adj.shape[0]==self.v_adj.shape[1]
        self.u_adj = torch.from_numpy(self.u_adj)
        self.v_adj = torch.from_numpy(self.v_adj)
        self.uv_adj = torch.from_numpy(self.uv_adj)
        self.dim_u = self.u_adj.shape[0]
        self.dim_v = self.v_adj.shape[0]
        self.size = (self.dim_u, self.dim_v)
        sign = f"{self.dataset_name} {self.mode} {self.n_splits} {self.v_idx} {self.u_idx} {self.split_zero}"
        root_dir = config.get("tmp_dir")
        super(BipartiteGraphDataset, self).__init__(root=os.path.join(root_dir, f"tmp-{self.seed}", sign), pre_filter=sign)

    @property
    def processed_file_names(self):
        return  [f"test_split_{i}.csv" for i in range(self.n_splits)]+\
                [f"train_pos_split_{i}.csv" for i in range(self.n_splits)]+\
                ["meta.mat"]

    def process(self):
        scio.savemat(os.path.join(self.processed_dir,"meta.mat"),
                     {"u_name":self.u_name,
                      "v_name":self.v_name,
                      "u_adj":self.u_adj,
                      "v_adj":self.v_adj,
                      "uv_adj":self.uv_adj,
                      })
        for i, (train_one_index, test_edge_index, test_edge_weight) in enumerate(self.split(self.uv_adj,
                                                                         n_splits=self.n_splits,
                                                                         mode=self.mode,
                                                                         u_idx=self.u_idx,
                                                                         v_idx=self.v_idx,
                                                                         shuffle=self.shuffle,
                                                                         split_zero=self.split_zero)):
            pd.DataFrame({"row":test_edge_index[0].numpy(),
                          "col":test_edge_index[1].numpy(),
                          "weight":test_edge_weight.numpy()}).to_csv(
                os.path.join(self.processed_dir, f"test_split_{i}.csv"),
                index=False)
            pd.DataFrame({"row":train_one_index[0].numpy(),
                          "col":train_one_index[1].numpy()}).to_csv(
                os.path.join(self.processed_dir, f"train_pos_split_{i}.csv"),
                index=False)



    def split(self, uv_adj, n_splits=1, mode="global", u_idx=None, v_idx=None, shuffle=False, split_zero=True):
        assert n_splits>=1
        assert mode in ("global", "local", "leave one")
        assert mode=="global" or not (u_idx is not None and v_idx is not None)
        print(f"dataset split mode:{mode}, u_idx:{u_idx}, v_idx:{v_idx}")
        uv_one_index = torch.stack(torch.where(uv_adj > 0.5))
        uv_zero_index = torch.stack(torch.where(uv_adj < 0.5))

        if mode=="global":
            yield from self.global_split_generator(uv_one_index, uv_zero_index, n_splits=n_splits,
                                                   shuffle=shuffle, split_zero=split_zero)
        elif mode=="local":
            yield from self.local_split_generator(uv_one_index, uv_zero_index, n_splits=n_splits,
                                                  u_idx=u_idx, v_idx=v_idx, shuffle=shuffle, split_zero=split_zero)
        elif mode=="leave one":
            if u_idx is True:
                for idx in range(self.uv_adj.shape[0]):
                    yield from self.local_split_generator(uv_one_index, uv_zero_index, 1, u_idx=idx, split_zero=split_zero)
            elif v_idx is True:
                for idx in range(self.uv_adj.shape[1]):
                    yield from self.local_split_generator(uv_one_index, uv_zero_index, 1, v_idx=idx, split_zero=split_zero)

    def global_split_generator(self, uv_one_index, uv_zero_index, n_splits, shuffle=False, split_zero=True):
        if n_splits==1:
            yield uv_one_index, torch.cat([uv_one_index, uv_zero_index], dim=1), torch.cat([torch.ones(uv_one_index.shape[1]),
                                                                                            torch.zeros(uv_zero_index.shape[1])],
                                                                                           dim=0)
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle)
            for (train_one_idx, test_one_idx), (train_zero_idx, test_zero_idx) in zip(
                    kf.split(uv_one_index.T), kf.split(uv_zero_index.T)):
                train_one_index = uv_one_index[:, train_one_idx]
                train_zero_index = uv_zero_index[:, train_zero_idx]
                test_one_index = uv_one_index[:, test_one_idx]
                test_zero_index = uv_zero_index[:, test_zero_idx]
                assert train_one_index.shape[1]+test_one_index.shape[1] == uv_one_index.shape[1]
                assert train_zero_index.shape[1]+test_zero_index.shape[1] == uv_zero_index.shape[1]
                if not split_zero:
                    test_zero_index = uv_zero_index
                test_edge_index = torch.cat([test_one_index, test_zero_index], dim=1)
                test_edge_weight = torch.zeros(test_edge_index.shape[1], dtype=torch.int)
                test_edge_weight[:test_one_index.shape[1]] = 1
                yield train_one_index, test_edge_index, test_edge_weight

    def local_split_generator(self, uv_one_index, uv_zero_index, n_splits,
                              u_idx=None, v_idx=None, shuffle=False, split_zero=True):
        if u_idx is not None:
            idx, dim = u_idx, 0
        else:
            idx, dim = v_idx, 1
        one_index = uv_one_index[:, uv_one_index[dim] == idx]
        remain_one_index = uv_one_index[:, uv_one_index[dim] != idx]
        zero_index = uv_zero_index[:, uv_zero_index[dim] == idx]
        remain_zero_index = uv_zero_index[:, uv_zero_index[dim] != idx]
        if n_splits==1:
            test_edge_index = torch.cat([one_index, zero_index], dim=1)
            test_edge_weight = torch.zeros(test_edge_index.shape[1], dtype=torch.int)
            test_edge_weight[:one_index.shape[1]] = 1
            yield remain_one_index, test_edge_index, test_edge_weight
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle)
            for (train_one_idx, test_one_idx), (train_zero_idx, test_zero_idx) in zip(
                    kf.split(one_index.T), kf.split(zero_index.T)):
                train_one_index = one_index[:, train_one_idx]
                train_zero_index = zero_index[:, train_zero_idx]
                test_one_index = one_index[:, test_one_idx]
                test_zero_index = zero_index[:, test_zero_idx]
                train_one_index = torch.cat([remain_one_index, train_one_index], dim=1)
                train_zero_index = torch.cat([remain_zero_index, train_zero_index], dim=1)

                if not split_zero:
                    test_zero_index = zero_index
                test_edge_index = torch.cat([test_one_index, test_zero_index], dim=1)
                test_edge_weight = torch.zeros(test_edge_index.shape[1], dtype=torch.int)
                test_edge_weight[:test_one_index.shape[1]] = 1
                yield train_one_index, test_edge_index, test_edge_weight

    def get(self, idx):
        test_file = os.path.join(self.processed_dir, f"test_split_{idx}.csv")
        train_file = os.path.join(self.processed_dir, f"train_pos_split_{idx}.csv")
        print(train_file, test_file)
        test_data = pd.read_csv(test_file)
        train_data = pd.read_csv(train_file)
        test_row_index = torch.from_numpy(test_data["row"].values)
        test_col_index = torch.from_numpy(test_data["col"].values)
        test_weight = torch.from_numpy(test_data["weight"].values).float()
        test_index = torch.stack([test_row_index, test_col_index])

        train_row_index = torch.from_numpy(train_data["row"].values)
        train_col_index = torch.from_numpy(train_data["col"].values)
        train_one_index = torch.stack([train_row_index, train_col_index])
        data = BipartiteGraph(u_adj=self.u_adj, u_x=self.u_adj,
                              v_adj=self.v_adj, v_x=self.v_adj,
                              uv_train_one_index=train_one_index,
                              uv_test_index=test_index,
                              uv_test_weight=test_weight,
                              uv_adj=self.uv_adj)
        data.split_v_adj(k=self.uk)
        data.split_u_adj(k=self.vk)
        u_sparse = data.u_edge_index.shape[1] / (data.u_adj.shape[0] ** 2)
        v_sparse = data.v_edge_index.shape[1] / (data.v_adj.shape[0] ** 2)
        uv_sparse = data.uv_train_one_index.shape[1] / data.u_adj.shape[0] / self.v_adj.shape[0]
        print(f"sparse u:{u_sparse}, v:{v_sparse}, uv:{uv_sparse}")
        return data

    def len(self):
        return self.n_splits


def select_topk(data, k=-1):
    if k<=0:
        return data
    assert k<=data.shape[1]
    val, col = torch.topk(data ,k=k)
    col = col.reshape(-1)
    row = torch.ones(1, k, dtype=torch.int)*torch.arange(data.shape[0]).view(-1, 1)
    row = row.view(-1)
    new_data = torch.zeros_like(data)
    new_data[row, col] = data[row, col]
    return new_data

def split_edge(adj, k=20, n_hop=1):
    val, col = torch.topk(adj, k=k * n_hop)
    row = torch.ones(1, k * n_hop, dtype=torch.int) * torch.arange(adj.shape[0]).view(-1, 1)
    vals = val.reshape(-1, n_hop, k).permute(1, 0, 2).reshape(n_hop, -1)
    rows = row.reshape(-1, n_hop, k).permute(1, 0, 2).reshape(n_hop, -1)
    cols = col.reshape(-1, n_hop, k).permute(1, 0, 2).reshape(n_hop, -1)
    non_zero_index = [torch.nonzero(val, as_tuple=False).view(-1) for val in vals]
    rows = [row[idx] for row, idx in zip(rows, non_zero_index)]
    cols = [col[idx] for col, idx in zip(cols, non_zero_index)]
    vals = [val[idx] for val, idx in zip(vals, non_zero_index)]
    edges_index = [torch.stack([row, col]) for row, col in zip(rows, cols)]
    return edges_index, vals
