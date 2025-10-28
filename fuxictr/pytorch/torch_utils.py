# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import sys
import os
import numpy as np
import torch
from torch import nn
import random
from functools import partial
import re

import torch.nn.functional as F
class MultiFeatureCosineContrastiveLoss(nn.Module):
    """
    支持多特征的余弦对比损失函数
    输入形状: v1, v2 - (batch_size, num_feat, embed_dim)
    输出: 对每个特征计算余弦距离后，在特征维度求和
    """
    def __init__(self, margin=0.2, reduction='mean'):
        """
        参数:
            margin: 边际值，负样本对需要达到的最小距离
            reduction: 损失 reduction方法 ('mean', 'sum', 'none')
        """
        super(MultiFeatureCosineContrastiveLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, v1, v2, labels):
        """
        前向传播
        
        参数:
            v1: 第一个embedding table选出的向量, shape: (batch_size, num_feat, embed_dim)
            v2: 第二个embedding table选出的向量, shape: (batch_size, num_feat, embed_dim)
            labels: 标签, 0或1, shape: (batch_size, 1)
            
        返回:
            loss: 计算得到的损失值
        """
        batch_size, num_feat, embed_dim = v1.shape
        # 1. 对向量进行L2归一化（在embed_dim维度）
        v1_norm = F.normalize(v1, p=2, dim=2)  # shape: (batch_size, num_feat, embed_dim)
        v2_norm = F.normalize(v2, p=2, dim=2)  # shape: (batch_size, num_feat, embed_dim)
        # 2. 计算余弦相似度（在embed_dim维度）
        # 使用torch.einsum进行批量矩阵乘法，计算每个特征对的余弦相似度
        cosine_sim = torch.einsum('bnd,bnd->bn', v1_norm, v2_norm)  # shape: (batch_size, num_feat)
        # 3. 将余弦相似度转换为余弦距离 (范围: 0-2)
        cosine_distance = 1 - cosine_sim  # shape: (batch_size, num_feat)
        # 4. 对特征维度进行求和
        summed_distance = cosine_distance.sum(dim=1)  # shape: (batch_size,)
        # 5. 根据标签计算损失
        # 扩展labels以匹配summed_distance的形状
        labels_expanded = labels.squeeze(1)  # shape: (batch_size,)
        # 当label=1时，使用余弦距离作为损失（拉近向量）
        positive_loss = labels_expanded * summed_distance
        # 当label=0时，使用max(0, margin - distance)作为损失（推开向量）
        negative_loss = (1 - labels_expanded) * torch.clamp(self.margin - summed_distance, min=0.0)
        # 6. 合并损失
        loss = positive_loss + negative_loss  # shape: (batch_size,)
        
        # 7. 根据reduction参数返回损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class CombinedLoss(nn.Module):
    """
    组合损失函数：BCELoss + reg_weight * CosineContrastiveLoss
    """
    def __init__(self, reg_weight=1.0, margin=0.2, reduction='mean'):
        """
        参数:
            reg_weight: 余弦对比损失的权重系数
            margin: 余弦对比损失的边际值
            reduction: 损失 reduction方法 ('mean', 'sum', 'none')
        """
        super(CombinedLoss, self).__init__()
        self.reg_weight = reg_weight
        
        # 初始化BCE损失函数
        self.bce_loss = nn.BCELoss(reduction=reduction)
        
        # 初始化自定义的余弦对比损失函数
        self.cosine_contrastive_loss = MultiFeatureCosineContrastiveLoss(margin=margin, reduction=reduction)
    
    def forward(self, probs, labels, reduction='mean'):
        """
        前向传播
        
        参数:
            probs: 模型输出的概率值，用于BCELoss，shape: (batch_size,) 范围[0,1]
            v1: 第一个embedding table选出的向量, shape: (batch_size, num_feat, embedding_dim)
            v2: 第二个embedding table选出的向量, shape: (batch_size, num_feat, embedding_dim)
            labels: 标签, 0或1, shape: (batch_size,)
            
        返回:
            total_loss: 组合损失值 = BCE损失 + reg_weight * 余弦对比损失
        """
        probs, v1, v2 = probs
        # 计算BCE损失
        bce_loss_value = self.bce_loss(probs, labels)
        
        # 计算余弦对比损失
        cosine_loss_value = self.cosine_contrastive_loss(v1, v2, labels)
        
        # 组合损失
        total_loss = bce_loss_value + self.reg_weight * cosine_loss_value
        
        return total_loss

    def get_loss_components(self, probs, v1, v2, labels):
        """
        返回各个损失分量，用于监控和分析
        """
        bce_loss_value = self.bce_loss(probs, labels)
        cosine_loss_value = self.cosine_contrastive_loss(v1, v2, labels)
        total_loss = bce_loss_value + self.reg_weight * cosine_loss_value
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss_value,
            'cosine_contrastive_loss': cosine_loss_value,
            'weighted_cosine_loss': self.reg_weight * cosine_loss_value
        }

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(gpu))
    else:
        device = torch.device("cpu")   
    return device

def get_optimizer(optimizer, params, lr):
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
    try:
        optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
    except:
        raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    return optimizer

def get_loss(loss):
    print("ephan0: ", loss, isinstance(loss, str))
    if isinstance(loss, str):
        if loss in ["bce", "binary_crossentropy", "binary_cross_entropy"]:
            loss = "binary_cross_entropy"
        elif "bce+ccl" in loss:
            reg_weight = float(loss.split('_')[1])
            margin = float(loss.split('_')[2])
            return CombinedLoss(reg_weight, margin)
    try:
        loss_fn = getattr(torch.functional.F, loss)
    except:
        try: 
            loss_fn = eval("losses." + loss)
        except:
            raise NotImplementedError("loss={} is not supported.".format(loss))       
    return loss_fn

def get_regularizer(reg):
    reg_pair = [] # of tuples (p_norm, weight)
    if isinstance(reg, float):
        reg_pair.append((2, reg))
    elif isinstance(reg, str):
        try:
            if reg.startswith("l1(") or reg.startswith("l2("):
                reg_pair.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
            elif reg.startswith("l1_l2"):
                l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
                reg_pair.append((1, float(l1_reg)))
                reg_pair.append((2, float(l2_reg)))
            else:
                raise NotImplementedError
        except:
            raise NotImplementedError("regularizer={} is not supported.".format(reg))
    return reg_pair

def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        if activation.lower() in ["prelu", "dice"]:
            assert type(hidden_units) == int
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "softmax":
            return nn.Softmax(dim=-1)
        elif activation.lower() == "prelu":
            return nn.PReLU(hidden_units, init=0.1)
        elif activation.lower() == "dice":
            from fuxictr.pytorch.layers.activations import Dice
            return Dice(hidden_units)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation

def get_initializer(initializer):
    if isinstance(initializer, str):
        try:
            initializer = eval(initializer)
        except:
            raise ValueError("initializer={} is not supported."\
                             .format(initializer))
    return initializer
