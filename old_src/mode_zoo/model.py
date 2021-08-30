import os
import time
import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as mf
from sklearn import metrics

class Model(pl.LightningModule):
    loger__ = logging.getLogger("__loger")
    loger__.propagate = False
    loger__.setLevel(logging.DEBUG)
    def __init__(self, lr=1e-4, log_interval=1, uk=20, vk=20, sigmoid_out=False, **kwargs):
        super(Model, self).__init__()
        self.lr = lr
        self.log_interval = log_interval
        self.model_dir = None
        self.uk = uk
        self.vk = vk
        self.sigmoid_out = sigmoid_out

    def step(self, batch, batch_id):
        """
        :param batch:
        :param batch_id:
        :return: (loss, u, v), info
        """
        return NotImplemented

    def training_step(self, batch, batch_id):
        (loss, score, u, v), info = self.step(batch, batch_id)
        writer = self.logger.experiment
        for key, val in info.items():
            writer.add_scalar(f"train/{key}", val, global_step=self.global_step)
            self.log(f"debug/{key}", val, prog_bar=True)
        if self.global_step%self.log_interval==0:
            with torch.no_grad():
                self.eval()
                (_, score, u, v), info = self.step(batch, batch_id)
                self.train()
                test_predict = score[batch.uv_test_edge_index[0], batch.uv_test_edge_index[1]]
                # train_u = u[batch.uv_edge_index[0]]
                # train_v = v[batch.uv_edge_index[1]]
                # test_u = u[batch.test_uv_edge_index[0]]
                # test_v = u[batch.test_uv_edge_index[1]]
                # train_predict = torch.sum(train_u*train_v, dim=-1)
                # test_predict = torch.sum(test_u*test_v, dim=-1)
                writer.add_histogram(tag="score", values=score, global_step=self.global_step)
                self.report(score, batch.uv_train_adj, tag="train")
                self.report(test_predict, batch.uv_test_edge_weight, tag="test")
        return loss

    # def validation_step(self, batch, batch_id=None):
    #     training = self.training
    #     self.eval()
    #     with torch.no_grad():
    #         (loss, score, u, v), info = self.step(batch, batch_id)
    #         writer = self.logger.experiment
    #         for key, val in info.items():
    #             writer.add_scalar(f"val/{key}", val, global_step=self.global_step)
    #             self.log(f"debug/{key}", val, prog_bar=True)
    #             test_predict = score[batch.uv_test_edge_index[0], batch.uv_test_edge_index[1]]
    #             writer.add_histogram(tag="val_score", values=score, global_step=self.global_step)
    #             self.report(score, batch.uv_train_adj, tag="val_train")
    #             self.report(test_predict, batch.uv_test_edge_weight, tag="val_test")
    #     self.train(training)

    def test_step(self, batch, batch_id=None, from_neighbor=False):
        self.eval()
        with torch.no_grad():
            (loss, score, u, v), info = self.step(batch, batch_id)
            # test_u = u[batch.test_uv_edge_index[0]]
            # test_v = u[batch.test_uv_edge_index[1]]
            # test_predict = torch.sum(test_u * test_v, dim=-1)
            if from_neighbor and u is not None and v is not None:
                merge_u = self.merge_neighbor_feature(batch.u_adj, u, self.uk)
                merge_v = self.merge_neighbor_feature(batch.v_adj, v, self.vk)
                if hasattr(self, "mask_u"):
                    mask_u = self.mask_u
                    mask_v = self.mask_v
                else:
                    u_idx, v_idx = batch.uv_train_one_index[0].unique(), batch.uv_train_one_index[1].unique()
                    mask_u = torch.zeros(u.shape[0], 1, dtype=torch.bool)
                    mask_u[u_idx] = True
                    mask_v = torch.zeros(v.shape[0], 1, dtype=torch.bool)
                    mask_v[v_idx] = True
                    self.register_buffer("mask_u", mask_u)
                    self.register_buffer("mask_v", mask_v)
                U = torch.where(mask_u, u, merge_u)
                V = torch.where(mask_v, v, merge_v)
                score = U @ V.T
                if self.sigmoid_out:
                    score = torch.sigmoid(score)

            test_predict = score[batch.uv_test_edge_index[0], batch.uv_test_edge_index[1]]
        self.sim_u = (u@u.T)/torch.norm(u, dim=-1, keepdim=True)/torch.norm(u.T, dim=0, keepdim=True)
        self.sim_v = (v@v.T)/torch.norm(v, dim=-1, keepdim=True)/torch.norm(v.T, dim=0, keepdim=True)
        self.logger.experiment.add_embedding(mat=torch.cat([u,v], dim=0),
                                             metadata=[f"r{i}" for i in range(u.shape[0])]+
                                                      [f"d{i}" for i in range(v.shape[0])],
                                             global_step=1)
        self.logger.experiment.add_embedding(mat=torch.cat([u, v], dim=0),
                                             metadata=[f"r" for i in range(u.shape[0])] +
                                                      [f"d" for i in range(v.shape[0])],
                                             global_step=0)
        if hasattr(self, "input_x"):
            self.logger.experiment.add_embedding(mat=self.input_x,
                                                 metadata=[f"r" for i in range(u.shape[0])] +
                                                          [f"d" for i in range(v.shape[0])],
                                                 global_step=2)
        return test_predict, batch.uv_test_edge_weight, batch.uv_test_edge_index

    def report(self, predict, target, tag="report", on_epoch=False):
        predict = predict.reshape(-1)
        target = target.reshape(-1)
        # auc = mf.auroc(predict, target)
        # aupr = mf.average_precision(predict, target)
        auc = metrics.roc_auc_score(y_score=predict, y_true=target)
        aupr = metrics.average_precision_score(y_score=predict, y_true=target)
        val, index = torch.topk(predict, k=100)
        top10 = target[index[:10]].mean()
        top50 = target[index[:50]].mean()
        top100 = target[index[:100]].mean()
        from pytorch_lightning.trainer.states import TrainerState
        if self.trainer and self.trainer._state!=TrainerState.FINISHED:
            self.log(f"{tag}/aupr", aupr, on_epoch=on_epoch, prog_bar=True)
            self.log(f"{tag}/auc", auc, on_epoch=on_epoch, prog_bar=True)
            self.log(f"{tag}/top10", top10, on_epoch=on_epoch, prog_bar=True)
            self.log(f"{tag}/top50", top50, on_epoch=on_epoch, prog_bar=True)
            self.log(f"{tag}/top100", top100, on_epoch=on_epoch, prog_bar=True)
        else:
            print(f"{tag} aupr:{aupr:.5f},auc:{auc:.5f}",
                  f"top10:{top10:.3f},top50:{top50:.3f},top100:{top100:.3f}")
        self.loger__.info(f"{tag} step:{self.current_epoch}, aupr:{aupr},auc:{auc},top10:{top10},top50:{top50},top100:{top100}")
        tensorboard = self.logger.experiment
        tensorboard.add_pr_curve(f"{tag}/pr_curve", target, predict,
                                 global_step=self.global_step)
        tensorboard.add_scalar(f"{tag}/aupr", aupr, global_step=self.global_step)
        tensorboard.add_scalar(f"{tag}/auc", auc, global_step=self.global_step)
        tensorboard.add_scalar(f"{tag}/top10", top10, global_step=self.global_step)
        tensorboard.add_scalar(f"{tag}/top50", top50, global_step=self.global_step)
        tensorboard.add_scalar(f"{tag}/top100", top100, global_step=self.global_step)

    def configure_optimizers(
            self,
    ):
        optimizer = optim.Adam(params=self.parameters(),
                               lr=self.lr,
                               weight_decay=1e-3)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 500], gamma=0.1)
        return optimizer

    def on_train_start(self) -> None:
        print("parameter nums:", self.num_parameters)
        if self.trainer and len(self.loger__.handlers)==0:
            log_dir = self.trainer.checkpoint_callback.dirpath
            format = '%Y-%m-%d %H-%M-%S'
            fm = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
                                   datefmt="%m/%d %H:%M:%S")
            save_dir = os.path.join(*(log_dir.split(os.path.sep)[:-3]))
            file_handler = logging.FileHandler(os.path.join(save_dir,
                                                            f"{time.strftime(format, time.localtime())}.log"))
            file_handler.setFormatter(fm)
            self.loger__.addHandler(file_handler)
            self.loger__.info(log_dir)
            print(self)
        log_dir = self.trainer.checkpoint_callback.dirpath
        self.model_dir = os.path.join(*(log_dir.split(os.path.sep)[:-3]))

    def info(self, message):
        self.loger__.info(message)

    def select_topk(self, data, k=-1):
        if k is None or k <= 0:
            return data
        assert k <= data.shape[1]
        val, col = torch.topk(data, k=k)
        col = col.reshape(-1)
        row = torch.ones(1, k, dtype=torch.int) * torch.arange(data.shape[0]).view(-1, 1)
        row = row.view(-1)
        new_data = torch.zeros_like(data)
        new_data[row, col] = data[row, col]
        return new_data

    def laplacian_matrix(self, S):
        x = torch.sum(S, dim=0)
        y = torch.sum(S, dim=1)
        L = 0.5*(torch.diag(x+y) - (S+S.T))  # neighborhood regularization matrix
        return L

    def merge_neighbor_feature(self, sims, features, k):
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

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def get_pos_weight(self, label):
        if not hasattr(self, "_pos_weight"):
            label = label.view(-1)
            pos_num = torch.nonzero(label, as_tuple=False).shape[0]
            total_num = label.shape[0]
            neg_num = total_num - pos_num
            pos_weight = neg_num / pos_num
            norm = total_num / (neg_num * 2)
            self.register_buffer("_pos_weight", torch.tensor(pos_weight, device=label.device))
            self.register_buffer("_norm", torch.tensor(norm, device=label.device))
            print("_pos_weight:", pos_weight, "_norm:", norm)
        return self._pos_weight

    def bce_loss_fn(self, predict, label, reduction="mean"):
        predict = predict.view(-1)
        label = label.view(-1)
        self.get_pos_weight(label)
        weight = self._pos_weight * label + 1 - label
        bce_loss = F.binary_cross_entropy(input=predict, target=label,
                                          weight=weight, reduction=reduction)
        return bce_loss

    def rank_loss_fn(self, pos_index, neg_index, score, margin=0.7, reduction='mean'):
        assert len(score.shape)==2 and len(pos_index.shape)==2
        one_idx = pos_index
        zero_idx = neg_index
        zero_mask = torch.randint(0, zero_idx.shape[1], (one_idx.shape[1],))
        zero_idx = zero_idx[:, zero_mask]
        pos = score[one_idx[0], one_idx[1]]
        neg = score[zero_idx[0], zero_idx[1]]
        rank_loss = F.margin_ranking_loss(pos, neg, target=torch.ones_like(pos),
                                          margin=margin, reduction=reduction)
        return rank_loss

    def graph_loss_fn(self, x, adj, topk=None, cache_name=None, reduction="mean"):
        if not hasattr(self, f"_{cache_name}") :
            adj = adj-torch.diag(torch.diag(adj))
            adj = self.select_topk(adj, k=topk)
            la = self.laplacian_matrix(adj)
            if cache_name:
                self.register_buffer(f"_{cache_name}", la)
        else:
            la = getattr(self, f"_{cache_name}")
            assert la.shape==adj.shape

        graph_loss = torch.trace(x.T@la@x)
        graph_loss = graph_loss/(x.shape[0]**2) if reduction=="mean" else graph_loss
        return graph_loss

    def mse_loss_fn(self, batch, score, reduction="mean"):
        label = batch.uv_train_adj
        self.get_pos_weight(label)
        pos_index = batch.uv_train_one_index
        neg_index = batch.uv_train_zero_index
        pos_score = score[pos_index[0], pos_index[1]]
        neg_score = score[neg_index[0], neg_index[1]]
        pos_loss = F.mse_loss(pos_score, batch.uv_train_one_weight, reduction=reduction)
        neg_loss = F.mse_loss(neg_score, batch.uv_train_zero_weight, reduction=reduction)
        loss = pos_loss*self._pos_weight+neg_loss
        return {"loss_mse":loss,
                "loss_mse_pos":pos_loss,
                "loss_mse_neg":neg_loss}

    def loss_all_fn(self, batch, score, u, v, u_alpha=0.0, v_alpha=0.0,
                    margin=0.7, reduction="mean"):
        label = batch.uv_train_adj
        pos_index = batch.uv_train_one_index
        neg_index = batch.uv_train_zero_index
        u_adj = batch.u_adj
        v_adj = batch.v_adj

        self.get_pos_weight(label)

        pos_score = score[pos_index[0], pos_index[1]]
        neg_score = score[neg_index[0], neg_index[1]]

        mse_pos_loss = F.mse_loss(pos_score.view(-1), batch.uv_train_one_weight.view(-1), reduction=reduction)
        mse_neg_loss = F.mse_loss(neg_score.view(-1), batch.uv_train_zero_weight.view(-1), reduction=reduction)
        mse_loss = mse_pos_loss * self._pos_weight + mse_neg_loss

        bce_loss = self.bce_loss_fn(predict=score, label=label, reduction=reduction)
        rank_loss = self.rank_loss_fn(pos_index=pos_index,
                                      neg_index=neg_index,
                                      score=score,
                                      margin=margin)
        u_graph_loss = self.graph_loss_fn(x=u, adj=u_adj, cache_name="ul",
                                          topk=self.uk,
                                          reduction=reduction)
        v_graph_loss = self.graph_loss_fn(x=v, adj=v_adj, cache_name="vl",
                                          topk=self.vk,
                                          reduction=reduction)
        graph_loss = u_graph_loss*u_alpha+v_graph_loss*v_alpha
        return {"loss_bce":bce_loss,
                "loss_rank":rank_loss,
                "loss_mse":mse_loss,
                "loss_mse_pos":mse_pos_loss,
                "loss_mse_neg":mse_neg_loss,
                "loss_graph":graph_loss,
                "loss_graph_u":u_graph_loss,
                "loss_graph_v":v_graph_loss}

    def init_bigraph(self, batch):
        """
        U  |  R    -->  U|VR
        RT |  V    --> UR|V
        """
        device = batch.u_edge_index.device
        size_u = batch.u_x.shape[0]
        self.u_edge_index = batch.u_edge_index
        self.u_edge_weight = batch.u_edge_weight
        self.v_edge_index = batch.v_edge_index + torch.tensor([[size_u],[size_u]], device=device)
        self.v_edge_weight = batch.v_edge_weight
        self.vr_edge_index = batch.uv_train_one_index + torch.tensor([[0],[size_u]], device=device)
        self.vr_edge_weight = batch.uv_train_one_weight
        self.ur_edge_index = reversed(batch.uv_train_one_index) + torch.tensor([[size_u],[0]], device=device)
        self.ur_edge_weight = batch.uv_train_one_weight
        self.full_edge_index = torch.cat([self.u_edge_index, self.ur_edge_index,
                                          self.vr_edge_index, self.v_edge_index], dim=1)
        self.full_edge_weight = torch.cat([self.u_edge_weight, self.ur_edge_weight,
                                           self.vr_edge_weight, self.v_edge_weight])


class SubModel(Model):
    def __init__(self, size_u, size_v, in_dim, input_type=None, **kwargs):
        """
        如果in_dim 为空， 通过input_type指定不可训练输入x的形式
        如果in_dim 不为空， 则输入x为随机可训练参数
        """
        super(SubModel, self).__init__(**kwargs)
        assert input_type in (None, "uv", "r", "uvr")
        assert in_dim and not input_type or not in_dim and input_type
        self.size_u = size_u
        self.size_v = size_v
        self.num_nodes = size_u + size_v
        self.use_embedding = True if in_dim else False
        self.in_dim = in_dim if self.use_embedding else self.num_nodes
        self.input_type = None if self.use_embedding else input_type
        self.build_input_x()
        margin = kwargs.get("margin", 0.7)
        u_alpha = kwargs.get("u_alpha", 0.25/2)
        v_alpha = kwargs.get("v_alpha", 0.125/2)
        self.register_buffer("margin", torch.tensor(margin))
        self.register_buffer("u_alpha", torch.tensor(u_alpha))
        self.register_buffer("v_alpha", torch.tensor(v_alpha))

    def step(self, batch, batch_id):
        x = self.build_input_x(batch=batch)
        score, u, v = self.forward(x, self.u_edge_index, self.u_edge_weight,
                                   self.v_edge_index, self.v_edge_weight,
                                   self.ur_edge_index, self.ur_edge_weight,
                                   self.vr_edge_index, self.vr_edge_weight)
        loss, loss_info = self.loss_fn(batch, score, u, v)
        loss_info["loss_total"] = loss
        return (loss, score, u, v), loss_info

    def forward(self, x, u_edge_index, u_edge_weight, v_edge_index, v_edge_weight,
                ur_edge_index, ur_edge_weight, vr_edge_index, vr_edge_weight):
        """
        :return: score, u, v
        """
        return NotImplemented

    def loss_fn(self, batch, score, u, v, reduction="mean"):
        all_loss = self.loss_all_fn(batch, score, u, v, margin=self.margin,
                                    u_alpha=self.u_alpha, v_alpha=self.v_alpha,
                                    reduction=reduction)
        return all_loss["loss_bce"], all_loss

    def build_input_x(self, batch=None):
        if not hasattr(self, "full_edge_index") and batch is not None:
            self.init_bigraph(batch)
        if hasattr(self, "input_x"):
            return self.input_x
        if self.use_embedding:
            self.input_x = nn.Parameter(torch.randn(self.num_nodes,
                                                    self.in_dim))
            # nn.init.kaiming_normal_(self.input_x)
            return self.input_x
        elif hasattr(self, "full_edge_index"):
            if self.input_type=="uv":
                edge_index = torch.cat([self.u_edge_index, self.v_edge_index], dim=1)
                edge_weight = torch.cat([self.u_edge_weight, self.v_edge_weight])
            elif self.input_type=="r":
                edge_index = torch.cat([self.ur_edge_index, self.vr_edge_index], dim=1)
                edge_weight = torch.cat([self.ur_edge_weight, self.vr_edge_weight])
            elif self.input_type=="uvr":
                edge_index = self.full_edge_index
                edge_weight = self.full_edge_weight
            else:
                raise NotImplemented
            from mode_zoo.gnn import gcn_norm
            x = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight,
                                        size=(self.num_nodes, self.num_nodes),
                                        device=self.device)
            # x = x.to_dense()
            norm_x = gcn_norm(edge_index=x, add_self_loops=False)[0].to_dense()
            x = norm_x*torch.norm(x)/torch.norm(norm_x)
            self.register_buffer("input_x", x)
            return self.input_x

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=0.1 * self.lr, weight_decay=1e-5)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * self.lr, max_lr=self.lr,
                                                   gamma=0.995, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return [optimizer], [lr_scheduler]
