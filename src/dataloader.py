import logging
import os
import torch
import pandas as pd
import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
import pytorch_lightning as pl

from . import DATA_TYPE_REGISTRY

class DRDataset():
    def __init__(self, dataset_name="Fdataset", drug_neighbor_num=15, disease_neighbor_num=15):
        assert dataset_name in ["Cdataset", "Fdataset", "lagcn", "lrssl", "hdvd"]
        self.dataset_name = dataset_name
        if dataset_name=="lrssl":
            old_data = load_DRIMC(name=dataset_name)
        elif dataset_name=="hdvd":
            old_data = load_HDVD()
        elif dataset_name=="lagcn":
            old_data = load_LAGCN()
        else:
            old_data = scio.loadmat(f"dataset/{dataset_name}.mat")

        self.drug_sim = old_data["drug"].astype(np.float)
        self.disease_sim = old_data["disease"].astype(np.float)
        self.drug_name = old_data["Wrname"].reshape(-1)
        self.drug_num = len(self.drug_name)
        self.disease_name = old_data["Wdname"].reshape(-1)
        self.disease_num = len(self.disease_name)
        self.interactions = old_data["didr"].T

        self.drug_edge = self.build_graph(self.drug_sim, drug_neighbor_num)
        self.disease_edge = self.build_graph(self.disease_sim, disease_neighbor_num)
        pos_num = self.interactions.sum()
        neg_num = np.prod(self.interactions.shape) - pos_num
        self.pos_weight = neg_num / pos_num
        print(f"dataset:{dataset_name}, drug:{self.drug_num}, disease:{self.disease_num}, pos weight:{self.pos_weight}")

    def build_graph(self, sim, num_neighbor):
        if num_neighbor>sim.shape[0] or num_neighbor<0:
            num_neighbor = sim.shape[0]
        neighbor = np.argpartition(-sim, kth=num_neighbor, axis=1)[:, :num_neighbor]
        row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
        col_index = neighbor.reshape(-1)
        edge_index = torch.from_numpy(np.array([row_index, col_index]).astype(int))
        values = torch.ones(edge_index.shape[1])
        values = torch.from_numpy(sim[row_index, col_index]).float()*values
        return (edge_index, values, sim.shape)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("dataset config")
        parser.add_argument("--dataset_name", default="Cdataset",
                                   choices=["Cdataset", "Fdataset", "lrssl", "hdvd", "lagcn"])
        parser.add_argument("--drug_neighbor_num", default=15, type=int)
        parser.add_argument("--disease_neighbor_num", default=15, type=int)
        return parent_parser



class Dataset():
    def __init__(self, dataset, mask, fill_unkown=True, stage="train", **kwargs):
        mask = mask.astype(bool)
        self.stage = stage
        self.one_mask = torch.from_numpy(dataset.interactions>0)
        row, col = np.nonzero(mask&dataset.interactions.astype(bool))
        self.valid_row = torch.tensor(np.unique(row))
        self.valid_col = torch.tensor(np.unique(col))
        if not fill_unkown:
            row_idx, col_idx = np.nonzero(mask)
            self.interaction_edge = torch.LongTensor([row_idx, col_idx]).contiguous()
            self.label = torch.from_numpy(dataset.interactions[mask]).float().contiguous()
            self.valid_mask = torch.ones_like(self.label, dtype=torch.bool)
            self.matrix_mask = torch.from_numpy(mask)
        else:
            row_idx, col_idx = torch.meshgrid(torch.arange(mask.shape[0]), torch.arange(mask.shape[1]))
            self.interaction_edge = torch.stack([row_idx.reshape(-1), col_idx.reshape(-1)])
            self.label = torch.clone(torch.from_numpy(dataset.interactions)).float()
            self.label[~mask] = 0
            self.valid_mask = torch.from_numpy(mask)
            self.matrix_mask = torch.from_numpy(mask)

        self.drug_edge = dataset.drug_edge
        self.disease_edge = dataset.disease_edge

        self.u_embedding = torch.from_numpy(dataset.drug_sim).float()
        self.v_embedding = torch.from_numpy(dataset.disease_sim).float()

        self.mask = torch.from_numpy(mask)
        pos_num = self.label.sum().item()
        neg_num = np.prod(self.mask.shape) - pos_num
        self.pos_weight = neg_num / pos_num

    def __str__(self):
        return f"{self.__class__.__name__}(shape={self.mask.shape}, interaction_num={len(self.interaction_edge)}, pos_weight={self.pos_weight})"

    @property
    def size_u(self):
        return self.mask.shape[0]

    @property
    def size_v(self):
        return self.mask.shape[1]

    def get_u_edge(self, union_graph=False):
        edge_index, value, size = self.drug_edge
        if union_graph:
            size = (self.size_u+self.size_v, )*2
        return edge_index, value, size

    def get_v_edge(self, union_graph=False):
        edge_index, value, size = self.disease_edge
        if union_graph:
            edge_index = edge_index + torch.tensor(np.array([[self.size_u], [self.size_u]]))
            size = (self.size_u + self.size_v,) * 2
        return edge_index, value, size

    def get_uv_edge(self, union_graph=False):
        train_mask = self.mask if self.stage=="train" else ~self.mask
        train_one_mask = train_mask & self.one_mask
        edge_index = torch.nonzero(train_one_mask).T
        value = torch.ones(edge_index.shape[1])
        size =  (self.size_u, self.size_v)
        if union_graph:
            edge_index = edge_index + torch.tensor([[0], [self.size_u]])
            size = (self.size_u + self.size_v,) * 2
        return edge_index, value, size

    def get_vu_edge(self, union_graph=False):
        edge_index, value, size = self.get_uv_edge(union_graph=union_graph)
        edge_index = reversed(edge_index)
        return edge_index, value, size

    def get_union_edge(self, union_type="u-uv-vu-v"):
        types = union_type.split("-")
        edges = []
        size = (self.size_u+self.size_v, )*2
        for type in types:
            assert type in ["u","v","uv","vu"]
            edge = self.__getattribute__(f"get_{type}_edge")(union_graph=True)
            edges.append(edge)
        edge_index = torch.cat([edge[0] for edge in edges], dim=1)
        value = torch.cat([edge[1] for edge in edges], dim=0)
        return edge_index, value, size

    @staticmethod
    def collate_fn(batch):
        return batch

# batch_size=1024*5 epochs 64 auroc:0.9357351384806536-aupr:0.553179962144391 Cdataset
# batch_size=1024*5 epochs 32 auroc:0.9353403407919936-aupr:0.5319780678531647 Cdataset
# batch_size=1024*10 epochs 128 auroc:0.9205757164088856-aupr:0.5228925211295569
# batch_size=1024*10 epochs 64 auroc:0.9161912302288697-aupr:0.4971947471119751
# batch_size=1024*10 epochs 32 auroc:0.8978203124764995-aupr:0.3906882262477967
# batch_size=1024*5 epochs 64 auroc:0.9214539160124139-aupr:0.5080356373187406
# batch_size=1024*5 epochs 32 auroc:0.9233284816950948-aupr:0.4833606718343333
# batch_size=1024*1 epochs 32 auroc:0.9190991473694987-aupr:0.44253258179201654
class GraphDataIterator(DataLoader):
    def __init__(self, dataset, mask, fill_unkown=True, stage="train", batch_size=1024*5, shuffle=False,
                 dataset_type="FullGraphDataset", **kwargs):
        # assert dataset_type in ["FullGraphDataset", "PairGraphDataset"]
        dataset_cls = DATA_TYPE_REGISTRY.get(dataset_type)
        dataset = dataset_cls(dataset, mask, fill_unkown, stage=stage, **kwargs)
        if len(dataset)<batch_size:
            logging.info(f"dataset size:{len(dataset)}, batch_size:{batch_size} is invalid!")
            batch_size = min(len(dataset), batch_size)
        if shuffle and stage=="train":
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        super(GraphDataIterator, self).__init__(dataset=dataset, batch_size=None, sampler=batch_sampler,
                                            collate_fn=Dataset.collate_fn, **kwargs)


class CVDataset(pl.LightningDataModule):
    """use for cross validation
       split_mode | n_splits |  drug_id   |   disease_id | description
       global     |   1      |   *        |     *        | case study
       global     |  10      |   *        |     *        | 10 fold
       local      |  -1      |   not None |     *        | local leave one for remove drug
       local      |  -1      |   None     |     not None | local leave one for remove disease
       local      |   1      |   int      |     *        | local leave one for remove specific drug
       local      |   1      |   None     |     int      | local leave one for remove specific drug
    """
    def __init__(self, dataset, split_mode="global", n_splits=10,
                 drug_idx=None, disease_idx=None, global_test_all_zero=False,
                 train_fill_unknown=True, seed=666, cached_dir="cached",
                 dataset_type="FullGraphDataset",
                 **kwargs):
        super(CVDataset, self).__init__()
        self.dataset = dataset
        self.split_mode = split_mode
        self.n_splits = n_splits
        self.global_test_all_zero = global_test_all_zero
        self.train_fill_unknown = train_fill_unknown
        self.seed = seed
        self.row_idx = drug_idx
        self.col_idx = disease_idx
        self.dataset_type = dataset_type
        self.save_dir = os.path.join(cached_dir, dataset.dataset_name,
                                     f"{self.split_mode}_{len(self)}_split_{self.row_idx}_{self.col_idx}")
        assert isinstance(n_splits, int) and n_splits>=-1

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("cross validation config")
        parser.add_argument("--split_mode", default="global", choices=["global", "local"])
        parser.add_argument("--n_splits", default=10, type=int)
        parser.add_argument("--drug_idx", default=None, type=int)
        parser.add_argument("--disease_idx", default=None, type=int)
        parser.add_argument("--global_test_all_zero", default=False, action="store_true", help="全局模式每折测试集是否测试所有未验证关联，默认：不测试")
        parser.add_argument("--train_fill_unknown", default=True, action="store_true", help="训练集中是否将测试集关联填0还是丢弃，默认：丢弃")
        parser.add_argument("--dataset_type", default=None, choices=["FullGraphDataset", "PairGraphDataset"])
        parser.add_argument("--seed", default=666, type=int)
        return parent_parser

    def fold_mask_iterator(self, interactions, mode="global", n_splits=10, row_idx=None, col_idx=None, global_test_all_zero=False, seed=666):
        assert mode in ["global", "local"]
        assert n_splits>=-1 and isinstance(n_splits, int)
        if mode=="global":
            if n_splits==1:
                mask = np.ones_like(interactions, dtype="bool")
                yield mask, mask
            else:
                kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                pos_row, pos_col = np.nonzero(interactions)
                neg_row, neg_col = np.nonzero(1 - interactions)
                assert len(pos_row) + len(neg_row) == np.prod(interactions.shape)
                for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                        kfold.split(neg_row)):
                    train_mask = np.zeros_like(interactions, dtype="bool")
                    test_mask = np.zeros_like(interactions, dtype="bool")
                    if global_test_all_zero:
                        test_neg_idx = np.arange(len(neg_row))
                    train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
                    train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
                    test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
                    test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
                    train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
                    test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
                    train_mask[train_edge[0], train_edge[1]] = True
                    test_mask[test_edge[0], test_edge[1]] = True
                    yield train_mask, test_mask
        elif mode=="local":
            if row_idx is not None:
                row_idxs = list(range(interactions.shape[0])) if n_splits==-1 else [row_idx]
                for idx in row_idxs:
                    yield self.get_fold_local_mask(interactions, row_idx=idx)
            elif col_idx is not None:
                col_idxs = list(range(interactions.shape[1])) if n_splits==-1 else [col_idx]
                for idx in col_idxs:
                    yield self.get_fold_local_mask(interactions, col_idx=idx)
        else:
            raise NotImplemented

    def get_fold_local_mask(self, interactions, row_idx=None, col_idx=None):
        train_mask = np.ones_like(interactions, dtype="bool")
        test_mask = np.zeros_like(interactions, dtype="bool")
        if row_idx is not None:
            train_mask[row_idx, :] = False
            test_mask[np.ones(interactions.shape[1], dtype="int")*row_idx,
                      np.arange(interactions.shape[1])] = True
        elif col_idx is not None:
            train_mask[:,col_idx] = False
            test_mask[np.arange(interactions.shape[0]),
                      np.ones(interactions.shape[0], dtype="int") * col_idx] = True
        return train_mask, test_mask

    def prepare_data(self):
        save_dir = self.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        import glob
        if len(glob.glob(os.path.join(save_dir, "split_*.mat")))!=len(self):
            for i, (train_mask, test_mask) in enumerate(self.fold_mask_iterator(interactions=self.dataset.interactions,
                                                                 mode=self.split_mode,
                                                                 n_splits=self.n_splits,
                                                                 global_test_all_zero=self.global_test_all_zero,
                                                                 row_idx=self.row_idx,
                                                                 col_idx=self.col_idx)):
                scio.savemat(os.path.join(save_dir, f"split_{i}.mat"),
                             {"train_mask":train_mask,
                              "test_mask":test_mask},
                             )

        data = scio.loadmat(os.path.join(self.save_dir, f"split_{self.fold_id}.mat"))
        self.train_mask = data["train_mask"]
        self.test_mask = data["test_mask"]

    def train_dataloader(self):
        return GraphDataIterator(self.dataset, self.train_mask, fill_unkown=self.train_fill_unknown,
                                 stage="train", dataset_type=self.dataset_type)

    def val_dataloader(self):
        return GraphDataIterator(self.dataset, self.test_mask, fill_unkown=True,
                                 stage="val", dataset_type=self.dataset_type)

    def __iter__(self):
        for fold_id in range(len(self)):
            self.fold_id = fold_id
            yield self

    def __len__(self):
        if self.split_mode=="global":
            return self.n_splits
        elif self.split_mode=="local":
            if self.n_splits==-1:
                if self.row_idx is not None:
                    return self.dataset.interactions.shape[0]
                elif self.col_idx is not None:
                    return self.dataset.interactions.shape[1]
            else:
                return 1

def load_DRIMC(root_dir="dataset/drimc", name="c", reduce=True):
    """ C drug:658, disease:409 association:2520 (False 2353)
        PREDICT drug:593, disease:313 association:1933 (Fdataset)
        LRSSL drug: 763, disease:681, association:3051
    """
    drug_chemical = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_chemical.txt"), sep="\t", index_col=0)
    drug_domain = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_domain.txt"), sep="\t", index_col=0)
    drug_go = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dc_go.txt"), sep="\t", index_col=0)
    disease_sim = pd.read_csv(os.path.join(root_dir, f"{name}_simmat_dg.txt"), sep="\t", index_col=0)
    if reduce:
        drug_sim =  (drug_chemical+drug_domain+drug_go)/3
    else:
        drug_sim = drug_chemical
    drug_disease = pd.read_csv(os.path.join(root_dir, f"{name}_admat_dgc.txt"), sep="\t", index_col=0).T
    if name=="lrssl":
        drug_disease = drug_disease.T
    rr = drug_sim.to_numpy(dtype=np.float32)
    rd = drug_disease.to_numpy(dtype=np.float32)
    dd = disease_sim.to_numpy(dtype=np.float32)
    rname = drug_sim.columns.to_numpy()
    dname = disease_sim.columns.to_numpy()
    return {"drug":rr,
            "disease":dd,
            "Wrname":rname,
            "Wdname":dname,
            "didr":rd.T}


def load_HDVD(root_dir="dataset/hdvd"):
    """drug:219, virus:34, association: 455"""
    dd = pd.read_csv(os.path.join(root_dir, "virussim.csv"), index_col=0).to_numpy(np.float32)
    rd = pd.read_csv(os.path.join(root_dir, "virusdrug.csv"), index_col=0)
    rr = pd.read_csv(os.path.join(root_dir, "drugsim.csv"), index_col=0).to_numpy(np.float32)
    rname = rd.index.to_numpy()
    dname = rd.columns.to_numpy()
    rd = rd.to_numpy(np.float32)
    return {"drug":rr,
            "disease":dd,
            "Wrname":rname,
            "Wdname":dname,
            "didr":rd.T}


def read_data(file):
    return pd.read_csv(file, header=None).to_numpy(dtype=np.float32)


def load_LAGCN(root_dir="dataset/lagcn"):
    """drug:598, disease:269 association:18416
    """
    dd = read_data(os.path.join(root_dir, "dis_sim.csv"))
    rd = read_data(os.path.join(root_dir, "drug_dis.csv"))
    rr = read_data(os.path.join(root_dir, "drug_sim.csv"))
    dname = np.arange(dd.shape[0])
    rname = np.arange(rr.shape[0])
    return {"drug":rr,
            "disease":dd,
            "Wrname":rname,
            "Wdname":dname,
            "didr":rd.T}
