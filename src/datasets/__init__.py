import os
import torch
import scipy.io as scio
import pandas as pd
import numpy as np

from datasets.dataset import BipartiteGraph, BipartiteGraphDataset

def read_data(file):
    return pd.read_csv(file, header=None).to_numpy(dtype=np.float32)

def load_BNNR(root_dir="data", name="Fdataset.mat"):
    """ Cdataset drug:663, disease:409 association:2532 (false 2352)
        Fdataset drug:593, disease:313 association: 1933
        DNdataset drug: 1490, disease:4516, association:1008
    """
    data = scio.loadmat(os.path.join(root_dir, name))
    rr = data["drug"].astype(np.float32)
    dd = data["disease"].astype(np.float32)
    rd = data["didr"].astype(np.float32).T
    dname = np.array([item[0] for item in data["Wdname"][:, 0]])
    rname = np.array([item[0] for item in data["Wrname"][:, 0]])
    return rr, dd, rd, rname, dname

def load_data(**kwargs):
    dataset_name = kwargs.get("dataset_name", "fdataset")
    # assert dataset_name in ("misim", "d1", "d2", "imcmda",
    #                         "fdataset", "cdataset", "dndataset",
    #                         "c", "lrssl", "predict", "lagcn",
    #                         "cdataset-snf", "lrssl-snf")
    from functools import partial
    loader = {
              "fdataset":partial(load_BNNR, name="Fdataset.mat"),
              }
    return BipartiteGraphDataset(config=kwargs, data_loader=loader[dataset_name])