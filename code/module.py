"""
@Author: hezf 
@Time: 2021/6/3 19:39 
@desc: 
"""
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import List


class MutiTaskLoss(nn.Module):
    def __init__(self, use_efficiency=False):
        """
        :param use_efficiency: 样本有效率标志，需要搭配起forward中的efficiency参数
        """
        super(MutiTaskLoss, self).__init__()
        self.use_efficiency = use_efficiency
        if use_efficiency:
            self.ans_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.ans_loss = nn.CrossEntropyLoss()
        self.prop_loss = FocalLoss(alpha=[0.759, 0.68, 0.9458, 0.9436, 0.9616, 0.915, 0.9618, 0.871, 0.8724, 0.994, 0.986, 0.994, 0.9986, 0.993, 0.991, 0.9968, 0.9986, 0.999, 0.9996],
                                   multi_label=True)
        # self.prop_loss = nn.BCELoss()
        if use_efficiency:
            self.entity_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.entity_loss = nn.CrossEntropyLoss()
        # 可学习的权重
        self.loss_weight = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, ans_true, ans_pred, prop_true, prop_pred, entity_true, entity_pred, efficiency=None):
        # 使用样本有效率
        if self.use_efficiency:
            assert efficiency is not None, 'efficiency is None'
            batch_size = ans_true.shape[0]
            loss1 = self.ans_loss(ans_pred, ans_true).unsqueeze(0).mm(efficiency.unsqueeze(1)).squeeze(0)/batch_size
            loss2 = self.prop_loss(prop_pred, prop_true.float(), efficiency)
            loss3 = self.entity_loss(entity_pred, entity_true).unsqueeze(0).mm(efficiency.unsqueeze(1)).squeeze(0)/batch_size
        else:
            loss1 = self.ans_loss(ans_pred, ans_true)
            loss2 = self.prop_loss(prop_pred, prop_true.float())
            loss3 = self.entity_loss(entity_pred, entity_true)
        weight = self.loss_weight.softmax(dim=0)
        loss = weight[0] * loss1 + weight[1]*loss2 + weight[2]*loss3
        return loss


class MutiTaskLossV1(nn.Module):
    def __init__(self, use_efficiency=False):
        """
        :param use_efficiency: 样本有效率标志，需要搭配起forward中的efficiency参数
        """
        super(MutiTaskLossV1, self).__init__()
        self.use_efficiency = use_efficiency
        if use_efficiency:
            self.ans_loss = nn.CrossEntropyLoss(reduction='none')
            self.entity_loss = nn.CrossEntropyLoss(reduction='none')
            self.method_loss = nn.CrossEntropyLoss(reduction='none')
            self.condition_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.ans_loss = nn.CrossEntropyLoss()
            self.entity_loss = nn.CrossEntropyLoss()
            self.method_loss = nn.CrossEntropyLoss()
            self.condition_loss = nn.CrossEntropyLoss()
        self.prop_loss = FocalLoss(alpha=[0.759, 0.68, 0.9458, 0.9436, 0.9616, 0.915, 0.9618, 0.871, 0.8724, 0.994, 0.986, 0.994, 0.9986, 0.993, 0.991, 0.9968, 0.9986, 0.999, 0.9996],
                                   multi_label=True)
        # self.prop_loss = nn.BCELoss()
        # 可学习的权重
        self.loss_weight = nn.Parameter(torch.ones(4), requires_grad=True)

    def forward(self, ans_true, ans_pred, prop_true, prop_pred, entity_true, entity_pred, method_true, method_pred, condition_true, condition_pred, efficiency=None):
        # 使用样本有效率
        if self.use_efficiency:
            assert efficiency is not None, 'efficiency is None'
            batch_size = ans_true.shape[0]
            loss1 = self.ans_loss(ans_pred, ans_true).unsqueeze(0).mm(efficiency.unsqueeze(1)).squeeze(0)/batch_size
            loss2 = self.prop_loss(prop_pred, prop_true.float(), efficiency)
            loss3 = self.entity_loss(entity_pred, entity_true).unsqueeze(0).mm(efficiency.unsqueeze(1)).squeeze(0)/batch_size
            loss4 = self.method_loss(method_pred, method_true).unsqueeze(0).mm(efficiency.unsqueeze(1)).squeeze(0)/batch_size
            loss5 = self.condition_loss(condition_pred, condition_true).unsqueeze(0).mm(efficiency.unsqueeze(1)).squeeze(0)/batch_size
        else:
            loss1 = self.ans_loss(ans_pred, ans_true)
            loss2 = self.prop_loss(prop_pred, prop_true.float())
            loss3 = self.entity_loss(entity_pred, entity_true)
            loss4 = self.method_loss(method_pred, method_true)
            loss5 = self.condition_loss(condition_pred, condition_true)
        weight = self.loss_weight.softmax(dim=0)
        loss = weight[0] * loss1 + weight[1]*loss2 + weight[2]*loss3 + weight[3]*(loss4 + loss5)/2
        return loss


class CrossEntropyWithEfficiency(nn.Module):
    def __init__(self):
        super(CrossEntropyWithEfficiency, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, true, pred, efficiency):
        batch_size = true.shape[0]
        loss = self.loss(pred, true).unsqueeze(0).mm(efficiency.unsqueeze(1)).squeeze(0) / batch_size
        return loss

# # rdrop 加入有效率前的备份
# class RDropLoss(nn.Module):
#     def __init__(self, alpha=4):
#         super(RDropLoss, self).__init__()
#         self.ans_loss = nn.CrossEntropyLoss()
#         self.prop_loss = FocalLoss(alpha=[0.759, 0.68, 0.9458, 0.9436, 0.9616, 0.915, 0.9618, 0.871, 0.8724, 0.994, 0.986, 0.994, 0.9986, 0.993, 0.991, 0.9968, 0.9986, 0.999, 0.9996],
#                                    multi_label=True)
#         self.entity_loss = nn.CrossEntropyLoss()
#         self.kl = nn.KLDivLoss(reduction='batchmean')
#         self.alpha = alpha
#         # 可学习的权重
#         self.loss_weight = nn.Parameter(torch.ones(3), requires_grad=True)
#
#     def r_drop(self, loss_func, pred: List, true, multi_label=True):
#         loss_0 = loss_func(pred[0], true)
#         loss_1 = loss_func(pred[1], true)
#         if multi_label:
#             kl_loss = (F.kl_div(pred[0].log(), pred[1], reduction='batchmean') + F.kl_div(pred[1].log(), pred[0], reduction='batchmean')) / 2
#         else:
#             kl_loss = (F.kl_div(F.log_softmax(pred[0], -1), F.softmax(pred[1], -1), reduction='batchmean') + F.kl_div(F.log_softmax(pred[1], -1), F.softmax(pred[0], -1), reduction='batchmean')) / 2
#         return loss_0 + loss_1 + self.alpha * kl_loss
#
#     def forward(self, ans_true, ans_pred: List, prop_true, prop_pred: List, entity_true, entity_pred: List):
#         loss1 = self.r_drop(self.ans_loss, ans_pred, ans_true, multi_label=False)
#         # loss1 = (self.ans_loss(ans_pred[0], ans_true) + self.ans_loss(ans_pred[1], ans_true))/2  # prop_only
#         loss2 = self.r_drop(self.prop_loss, prop_pred, prop_true.float(), multi_label=True)
#         loss3 = self.r_drop(self.entity_loss, entity_pred, entity_true, multi_label=False)
#         # loss3 = (self.entity_loss(entity_pred[0], entity_true) + self.entity_loss(entity_pred[1], entity_true))/2  # prop_only
#         weight = self.loss_weight.softmax(dim=0)
#         loss = weight[0] * loss1 + weight[1]*loss2 + weight[2]*loss3
#         return loss


class RDropLoss(nn.Module):
    def __init__(self, alpha=4):
        super(RDropLoss, self).__init__()
        self.ans_loss = nn.CrossEntropyLoss(reduction='none')
        self.prop_loss = FocalLoss(alpha=[0.759, 0.68, 0.9458, 0.9436, 0.9616, 0.915, 0.9618, 0.871, 0.8724, 0.994, 0.986, 0.994, 0.9986, 0.993, 0.991, 0.9968, 0.9986, 0.999, 0.9996],
                                   multi_label=True)
        self.entity_loss = nn.CrossEntropyLoss(reduction='none')
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.alpha = alpha
        # 可学习的权重
        self.loss_weight = nn.Parameter(torch.ones(3), requires_grad=True)

    def r_drop(self, loss_func, pred: List, true, efficiency, multi_label=True):
        if efficiency is None:
            efficiency = torch.tensor([1.0] * pred[0].shape[0])
        # 多标签时，是用focalloss
        if multi_label:
            loss_0 = loss_func(pred[0], true, efficiency)
            loss_1 = loss_func(pred[1], true, efficiency)
        else:
            batch_size = pred[0].shape[0]
            loss_0 = loss_func(pred[0], true).unsqueeze(0).mm(efficiency.unsqueeze(1)).squeeze(0)/batch_size
            loss_1 = loss_func(pred[1], true).unsqueeze(0).mm(efficiency.unsqueeze(1)).squeeze(0)/batch_size
        if multi_label:
            kl_loss = (F.kl_div(pred[0].log(), pred[1], reduction='batchmean') + F.kl_div(pred[1].log(), pred[0], reduction='batchmean')) / 2
        else:
            kl_loss = (F.kl_div(F.log_softmax(pred[0], -1), F.softmax(pred[1], -1), reduction='batchmean') + F.kl_div(F.log_softmax(pred[1], -1), F.softmax(pred[0], -1), reduction='batchmean')) / 2
        return loss_0 + loss_1 + self.alpha * kl_loss

    def forward(self, ans_true, ans_pred: List, prop_true, prop_pred: List, entity_true, entity_pred: List, efficiency):
        loss1 = self.r_drop(self.ans_loss, ans_pred, ans_true, efficiency, multi_label=False)
        loss2 = self.r_drop(self.prop_loss, prop_pred, prop_true.float(), efficiency, multi_label=True)
        loss3 = self.r_drop(self.entity_loss, entity_pred, entity_true, efficiency, multi_label=False)
        weight = self.loss_weight.softmax(dim=0)
        loss = weight[0] * loss1 + weight[1]*loss2 + weight[2]*loss3
        return loss


class MutiTaskLossFocal(nn.Module):
    def __init__(self):
        super(MutiTaskLossFocal, self).__init__()
        self.ans_loss = FocalLoss()
        self.prop_loss = FocalLoss(alpha=[0.759, 0.68, 0.9458, 0.9436, 0.9616, 0.915, 0.9618, 0.871, 0.8724, 0.994, 0.986, 0.994, 0.9986, 0.993, 0.991, 0.9968, 0.9986, 0.999, 0.9996],
                                   multi_label=True)
        self.entity_loss = FocalLoss()
        # 可学习的权重
        self.loss_weight = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, ans_true, ans_pred, prop_true, prop_pred, entity_true, entity_pred):
        loss1 = self.ans_loss(ans_pred, ans_true)
        loss2 = self.prop_loss(prop_pred, prop_true.float())
        loss3 = self.entity_loss(entity_pred, entity_true)
        weight = self.loss_weight.softmax(dim=0)
        loss = weight[0] * loss1 + weight[1]*loss2 + weight[2]*loss3
        return loss


class Metric(object):
    def __init__(self):
        super(Metric, self).__init__()

    def calculate(self, ans_pred, ans_true, prop_pred, prop_true, entity_pred, entity_true):
        ans_f1 = f1_score(ans_true, ans_pred, average='micro')
        # ans_matrix = confusion_matrix(ans_true, ans_pred)
        prop_j = jaccard_score(prop_true, prop_pred, average='micro')
        entity_f1 = f1_score(entity_true, entity_pred, average='micro')
        return ans_f1, prop_j, entity_f1


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        x = self.linear(hidden_states).squeeze(-1)
        x = x.masked_fill(mask, -np.inf)
        attention_value = x.softmax(dim=-1).unsqueeze(1)
        x = torch.bmm(attention_value, hidden_states).squeeze(1)
        return x


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1.0, size_average=True, multi_label=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(self.alpha, list):
            self.alpha = torch.tensor(self.alpha)
        self.size_average = size_average
        self.multi_label = multi_label

    def forward(self, logits, labels, efficiency=None):
        """
        logits: batch_size * n_class
        labels: batch_size
        """
        batch_size, n_class = logits.shape[0], logits.shape[1]
        # 多分类分类
        if not self.multi_label:
            one_hots = torch.zeros([batch_size, n_class]).to(logits.device).scatter_(-1, labels.unsqueeze(-1), 1)
            p = torch.nn.functional.softmax(logits, dim=-1)
            log_p = torch.log(p)
            loss = - one_hots * (self.alpha * ((1 - p) ** self.gamma) * log_p)
        # 多标签分类
        else:
            p = logits
            pt = (labels - (1 - p)) * (2*labels-1)
            if isinstance(self.alpha, float):
                alpha_t = (labels - (1 - self.alpha)) * (2*labels-1)
            else:
                alpha_t = (labels - (1 - self.alpha.to(logits.device))) * (2*labels-1)
            loss = - alpha_t * ((1 - pt)**self.gamma) * torch.log(pt)
        # 加入有效率的计算
        if efficiency is not None:
            loss = torch.diag_embed(efficiency).mm(loss)
        if self.size_average:
            return loss.sum()/batch_size
        else:
            return loss.sum()


class FGM(object):
    """
    对抗攻击
    """
    def __init__(self, model: nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}
