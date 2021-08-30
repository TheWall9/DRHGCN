import os
import torch
import numpy as np
import pandas as pd
from sklearn import metrics
import scipy.io as scio
from collections import defaultdict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import clac_metric

def build_model(config, dataset):
    model = config["model"](dataset.dim_u, dataset.dim_v, **config)
    log_dir = os.path.join("runs",
                            config["dataset_name"],
                            config["comment"],
                            model.__class__.__name__)
    checkpoint_callback = ModelCheckpoint(monitor="test/auc",
                                          mode="max",
                                          save_top_k=1,
                                          verbose=False,
                                          save_last=True)
    trainer = pl.Trainer(max_epochs=config["max_epochs"],
                         fast_dev_run=config["fast_dev_run"],
                         profiler=config["profiler"],
                         default_root_dir=log_dir,
                         checkpoint_callback=checkpoint_callback,
                         gpus=config.get("gpus",None))
    trainer.fit(model, train_dataloader=dataset)
    return model, model.model_dir


def anaylse_result(predict, label, edge_index, dataset, split_id, logger=None, save_dir=".", save=False, tag="final",
                   ):
    # global topk
    edge_num = edge_index.shape[1]
    ks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 400, 500, 800, 1000, 1200, 1500, 2000]
    ks = ks[:sum(np.array(ks)<=edge_num)]
    val, index = torch.topk(predict, k=max(ks))
    one_num = len(torch.nonzero(label, as_tuple=False))
    num = []
    for k in ks:
        val = label[index[:k]].sum().item()
        num.append(val)
    top_val = pd.DataFrame({"topk":ks, "num":num})
    top_val["total"] = np.where(top_val["topk"]<one_num, top_val["topk"], one_num)
    top_val["rate"] = top_val["num"]/top_val["total"]

    # global auroc, aupr
    fpr, tpr, _ = metrics.roc_curve(y_true=label, y_score=predict)
    roc_auc = metrics.auc(fpr, tpr)
    ap_auc = metrics.average_precision_score(y_true=label, y_score=predict)

    split_id_set = torch.unique(split_id)
    eval_res = defaultdict(list)
    eval_res["split_id"].append("all")
    eval_res["aupr"].append(ap_auc)
    eval_res["auroc"].append(roc_auc)
    lagcn_metric = clac_metric.evaluate(label=label, predict=predict, is_final=(tag=="final"))

    for key, val in lagcn_metric.items():
        eval_res[key].append(val)

    if len(split_id_set)!=1:
        for idx in split_id_set.numpy():
            mask = split_id==idx
            split_label = label[mask]
            split_predict = predict[mask]
            split_roc_auc = metrics.roc_auc_score(y_true=split_label,
                                                  y_score=split_predict)
            split_ap_auc = metrics.average_precision_score(y_true=split_label,
                                                           y_score=split_predict)
            lagcn_metric = clac_metric.evaluate(label=split_label, predict=split_predict)
            for key, val in lagcn_metric.items():
                eval_res[key].append(val)
            eval_res["aupr"].append(split_ap_auc)
            eval_res["auroc"].append(split_roc_auc)
            eval_res["split_id"].append(idx)
    metric = pd.DataFrame(eval_res)
    if logger is not None:
        logger.info(f"\n{top_val}")
        logger.info(f"\n{metric}")
        logger.info(f"{tag} analyse result: aupr:{ap_auc}, auroc:{roc_auc}")
    print(top_val)
    print(metric)
    print(f"{tag} analyse result: aupr:{ap_auc}, auroc:{roc_auc}")

    if save:
        save_file = os.path.join(save_dir, f"{tag}_result_roc_{roc_auc:.4f}.xlsx")
        writer = pd.ExcelWriter(save_file)
        metric.to_excel(writer, index=False, sheet_name="metric")
        scio.savemat(os.path.join(save_dir, f"score_{tag}.mat"),
                     {"score": predict.cpu().numpy(),
                      "label": label.cpu().numpy(),
                      "row": edge_index[0].cpu().numpy(),
                      "col": edge_index[1].cpu().numpy(),
                      "split_id": split_id.cpu().numpy(),
                      })
        size = dataset.size
        if tag!="final" and edge_index.shape[1]<np.prod(size):
            predict_matrix = torch.sparse_coo_tensor(indices=edge_index,
                                                     values=predict,
                                                     size=size).to_dense()
            label_matrix = torch.sparse_coo_tensor(indices=edge_index,
                                                   values=label,
                                                   size=size).to_dense()
            full_label_matrix = dataset.uv_adj
            row_top_val = pd.DataFrame({"name":dataset.u_name,
                                        "total_adj_num":full_label_matrix.sum(dim=1),
                                        "test_adj_num":label_matrix.sum(dim=1),
                                        "sim_weight":dataset.u_adj.sum(dim=1)})
            col_top_val = pd.DataFrame({"name":dataset.v_name,
                                        "total_adj_num": full_label_matrix.sum(dim=0),
                                        "test_adj_num": label_matrix.sum(dim=0),
                                        "sim_weight": dataset.v_adj.sum(dim=1)})
            # local aupr, auroc
            aps = []
            rocs = []
            for predict_row, predict_label in zip(predict_matrix, label_matrix):
                if predict_label.sum()>0:
                    aps.append(metrics.average_precision_score(y_score=predict_row,
                                                               y_true=predict_label))
                    rocs.append(metrics.roc_auc_score(y_score=predict_row,
                                                      y_true=predict_label))
                else:
                    aps.append(np.NAN)
                    rocs.append(np.NAN)
            row_top_val["aupr"] = aps
            row_top_val["auroc"] = rocs

            aps = []
            rocs = []
            for predict_row, predict_label in zip(predict_matrix.T, label_matrix.T):
                if predict_label.sum() > 0:
                    aps.append(metrics.average_precision_score(y_score=predict_row,
                                                               y_true=predict_label))
                    rocs.append(metrics.roc_auc_score(y_score=predict_row,
                                                      y_true=predict_label))
                else:
                    aps.append(np.NAN)
                    rocs.append(np.NAN)
            col_top_val["aupr"] = aps
            col_top_val["auroc"] = rocs

            # local topk
            ks = [1, 2, 3, 5, 10, 20, 50, 100]
            ks = ks[:sum(np.array(ks)<=edge_num)]
            row_grid = torch.arange(size[0]).view(-1, 1).repeat(1, size[1])
            col_grid = torch.arange(size[1]).view(1, -1).repeat(size[0], 1)

            val, col = torch.topk(predict_matrix, k=max(ks), dim=1)
            val, row = torch.topk(predict_matrix, k=max(ks), dim=0)
            for k in ks:
                row_idx = row_grid[:, :k].reshape(-1)
                col_idx = col[:, :k].reshape(-1)
                val = label_matrix[row_idx, col_idx].view(-1, k).sum(dim=-1)
                row_top_val[f"top_{k}"] = val.numpy()

                row_idx = row[:k,:].reshape(-1)
                col_idx = col_grid[:k,:].reshape(-1)
                val = label_matrix[row_idx, col_idx].view(k, -1).sum(dim=0)
                col_top_val[f"top_{k}"] = val.numpy()

            for k in ks:
                num = np.where(k<row_top_val["test_adj_num"],
                               k,
                               row_top_val["test_adj_num"])
                row_top_val[f"rate_top_{k}"] = (row_top_val[f"top_{k}"]*1.0/num).fillna(1)
                num = np.where(k< col_top_val["test_adj_num"],
                               k,
                               col_top_val["test_adj_num"])
                col_top_val[f"rate_top_{k}"] = (col_top_val[f"top_{k}"]*1.0/num).fillna(1)


            top_val.to_excel(writer, sheet_name="topk", index=False)
            row_top_val.to_excel(writer, sheet_name="drug")
            col_top_val.to_excel(writer, sheet_name="disease")
        writer.close()
        if logger is not None:
            logger.info(f"{tag} {save_file}")
        print(f"{tag} {save_file}")