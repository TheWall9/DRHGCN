import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn import metrics


class BaseModel(pl.LightningModule):
    DATASET_TYPE: None

    def __init__(self):
        super(BaseModel, self).__init__()

    def select_topk(self, data, k=-1):
        if k is None or k <= 0:
            return data
        assert k <= data.shape[1]
        val, col = torch.topk(data, k=k)
        col = col.reshape(-1)
        row = torch.ones(1, k, dtype=torch.int) * torch.arange(data.shape[0]).view(-1, 1)
        row = row.view(-1).to(device=data.device)
        new_data = torch.zeros_like(data)
        new_data[row, col] = data[row, col]
        return new_data

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

    def neighbor_smooth(self, sims, features, replace_rate=0.2):
        merged_u = self.merge_neighbor_feature(sims, features)
        mask = torch.rand(merged_u.shape[0], device=sims.device)
        mask = torch.floor(mask + replace_rate).type(torch.bool)
        new_features = torch.where(mask, merged_u, features)
        return new_features

    def laplacian_matrix(self, S):
        x = torch.sum(S, dim=0)
        y = torch.sum(S, dim=1)
        L = 0.5*(torch.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L

    def graph_loss_fn(self, x, edge, topk=None, cache_name=None, reduction="mean"):
        if not hasattr(self, f"_{cache_name}") :
            adj = torch.sparse_coo_tensor(*edge).to_dense()
            adj = adj-torch.diag(torch.diag(adj))
            adj = self.select_topk(adj, k=topk)
            la = self.laplacian_matrix(adj)
            if cache_name:
                self.register_buffer(f"_{cache_name}", la)
        else:
            la = getattr(self, f"_{cache_name}")
            assert la.shape==edge[2]

        graph_loss = torch.trace(x.T@la@x)
        graph_loss = graph_loss/(x.shape[0]**2) if reduction=="mean" else graph_loss
        return graph_loss

    def mse_loss_fn(self, predict, label, pos_weight):
        predict = predict.view(-1)
        label = label.view(-1)
        pos_mask = label>0
        loss = F.mse_loss(predict, label, reduction="none")
        loss_pos = loss[pos_mask].mean()
        loss_neg = loss[~pos_mask].mean()
        loss_mse = loss_pos*pos_weight+loss_neg
        return {"loss_mse":loss_mse,
                "loss_mse_pos":loss_pos,
                "loss_mse_neg":loss_neg,
                "loss":loss_mse}

    def bce_loss_fn(self, predict, label, pos_weight):
        predict = predict.reshape(-1)
        label = label.reshape(-1)
        weight = pos_weight * label + 1 - label
        loss = F.binary_cross_entropy(input=predict, target=label, weight=weight)
        return {"loss_bce":loss,
                "loss":loss}

    def focal_loss_fn(self, predict, label, alpha, gamma):
        predict = predict.view(-1)
        label = label.view(-1)
        ce_loss = F.binary_cross_entropy(
            predict, label, reduction="none"
        )
        p_t = predict*label+(1-predict)*(1-label)
        loss = ce_loss*((1-p_t)**gamma)
        alpha_t = alpha * label + (1-alpha)*(1-label)
        focal_loss = (alpha_t * loss).mean()
        return {"loss_focal":focal_loss,
                "loss":focal_loss}

    def rank_loss_fn(self, predict, label, margin=0.8, reduction='mean'):
        predict = predict.reshape(-1)
        label = label.reshape(-1)
        pos_mask = label > 0
        pos = predict[pos_mask]
        neg = predict[~pos_mask]
        neg_mask = torch.randint(0, neg.shape[0], (pos.shape[0],), device=label.device)
        neg = neg[neg_mask]

        rank_loss = F.margin_ranking_loss(pos, neg, target=torch.ones_like(pos),
                                          margin=margin, reduction=reduction)
        return {"loss_rank":rank_loss,
                "loss":rank_loss}

    def get_epoch_auroc_aupr(self, outputs):
        predict = [output["predict"].detach() for output in outputs]
        label = [output["label"] for output in outputs]
        predict = torch.cat(predict).cpu().view(-1)
        label = torch.cat(label).cpu().view(-1)
        aupr = metrics.average_precision_score(y_true=label, y_score=predict)
        auroc = metrics.roc_auc_score(y_true=label, y_score=predict)
        return auroc, aupr

    def get_epoch_loss(self, outputs):
        loss_keys = [key for key in outputs[0] if key.startswith("loss")]
        loss_info = {key: [output[key].detach().cpu() for output in outputs if not torch.isnan(output[key])] for key in loss_keys}
        loss_info = {key: sum(value)/len(value) for key, value in loss_info.items()}
        return loss_info

    def training_epoch_end(self, outputs):
        stage = "train"
        loss_info = self.get_epoch_loss(outputs)
        auroc, aupr = self.get_epoch_auroc_aupr(outputs)
        # self.log(f"{stage}/loss", loss_info["loss"], prog_bar=True)
        self.log(f"{stage}/auroc", auroc, prog_bar=True)
        self.log(f"{stage}/aupr", aupr, prog_bar=True)
        writer = self.logger.experiment
        for key, value in loss_info.items():
            writer.add_scalar(f"{stage}_epoch/{key}", value, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/auroc", auroc, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/aupr", aupr, global_step=self.current_epoch)

    def validation_epoch_end(self, outputs):
        stage = "val"
        loss_info = self.get_epoch_loss(outputs)
        auroc, aupr = self.get_epoch_auroc_aupr(outputs)
        # self.log(f"{stage}/loss", loss_info["loss"], prog_bar=True)
        self.log(f"{stage}/auroc", auroc, prog_bar=True)
        self.log(f"{stage}/aupr", aupr, prog_bar=True)
        writer = self.logger.experiment
        for key, value in loss_info.items():
            writer.add_scalar(f"{stage}_epoch/{key}", value, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/auroc", auroc, global_step=self.current_epoch)
        writer.add_scalar(f"{stage}_epoch/aupr", aupr, global_step=self.current_epoch)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items