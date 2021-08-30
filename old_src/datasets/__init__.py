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

def load_HDVD(root_dir="datasets/HDVD"):
    """drug:219, virus:34, association: 455"""
    dd = pd.read_csv(os.path.join(root_dir, "virussim.csv"), index_col=0).to_numpy(np.float32)
    rd = pd.read_csv(os.path.join(root_dir, "virusdrug.csv"), index_col=0)
    rr = pd.read_csv(os.path.join(root_dir, "drugsim.csv"), index_col=0).to_numpy(np.float32)
    rname = rd.index.to_numpy()
    dname = rd.columns.to_numpy()
    rd = rd.to_numpy(np.float32)
    return rr, dd, rd, rname, dname

def load_LAGCN(root_dir="datasets/Ldataset"):
    """drug:598, disease:269 association:18416
    """
    dd = read_data(os.path.join(root_dir, "dis_sim.csv"))
    rd = read_data(os.path.join(root_dir, "drug_dis.csv"))
    rr = read_data(os.path.join(root_dir, "drug_sim.csv"))
    dname = np.arange(dd.shape[0])
    rname = np.arange(rr.shape[0])
    return rr, dd, rd, rname, dname

def load_DRIMC(root_dir="datasets/LRSSL", name="lrssl", reduce=True):
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
