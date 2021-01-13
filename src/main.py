import torch
import pytorch_lightning as pl
from datasets import load_data
from utils import build_model, anaylse_result
from mode_zoo.exp import DRHGCN

def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description="DRHGCN")
    parser.add_argument("-d", "--dataset_name", default="fdataset", type=str,
                        choices=["c", "fdataset", "cdataset", "lrssl", "predict", "lagcn"])
    parser.add_argument("-c", "--comment", default="debug")
    parser.add_argument("-m", "--mode", default="global", type=str, choices=["global", "local", "leave one"])
    parser.add_argument("-n", "--n_splits", default=5, type=int, choices=[5, 10, -1], help="cross valid fold num")
    parser.add_argument("-t", "--times", default=-1, type=int, choices=[-1, 5, 10], help="fold id, -1 means all fold")
    parser.add_argument("--u_idx", default=None, help="when split mode is local or leave one, test when drug u_idx is removed")
    parser.add_argument("--v_idx", default=True, help="when split mode is local or leave one, test when drug u_idx is removed")
    parser.add_argument("--tmp_dir", default="tmp")
    parser.add_argument("--save", default=True, help="whether save predict results")
    parser.add_argument("--split_zero", default=True, help="whether split zero when split test data")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--uk", default=15, type=int, help="u topk")
    parser.add_argument("--vk", default=15, type=int, help="v topk")
    parser.add_argument("--from_neighbor", default=True, type=bool)
    parser.add_argument("--fast_dev_run", default=False, type=bool)
    parser.add_argument("--profiler", default=False, type=bool)
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--max_epochs", default=400, type=int)
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument("--embedding_dim", default=64, type=int)
    parser.add_argument("--layer_num", default=3, type=int)
    parser.add_argument("--lr", default=0.05, type=float)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--edge_dropout", default=0.2, type=float)
    parser.add_argument("--gnn_mode", default="gcnt", choices=["gcn","gcna","gcnt","a","t"])
    parser.add_argument("--model", default="DRHGCN", help="model class name",)
                        # choices=["DRHGCN", "InterGCN", "IntraGCN", "MixGCN"])
    args = parser.parse_args()
    args.model = globals()[args.model]
    return args

def eval(config):
    pl.seed_everything(config["seed"])
    config["comment"] = f"{config['comment']}-seed-{config['seed']}"
    config["n_splits"] = 99999 if config["times"]<0 and config["mode"]=="leave one" else config["n_splits"]
    config["times"] = config["times"] if config["times"]>=0 else config["n_splits"]
    dataset = load_data(**config)
    predicts = []
    labels = []
    edges_index = []
    split_idxs = []
    for i, data in zip(range(config["times"]), dataset):
        if config["times"]!=config["n_splits"] and config["times"]!=i+1:
            continue
        model, log_dir = build_model(config, data)
        predict, label, edge_index = model.test_step(data, from_neighbor=config["from_neighbor"])
        predicts.append(predict)
        labels.append(label)
        edges_index.append(edge_index)
        split_idxs.append(torch.ones(len(predict), dtype=torch.int)*i)
        anaylse_result(predict, label, edge_index,
                       dataset, split_idxs[-1], model, log_dir,
                       save=config.get("save", False),
                       tag=f"split_{i}")
        model.info(f"split {i} end")
    model.info(f"{model}")
    model.info(f"{config}")
    predicts = torch.cat(predicts)
    labels = torch.cat(labels)
    edges_index = torch.cat(edges_index, dim=-1)
    split_idxs = torch.cat(split_idxs)


if __name__=="__main__":
    args = get_parser()
    config = vars(args)
    print(config)
    eval(config)
    # analyse(config)