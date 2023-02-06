#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

# reference: https://arxiv.org/abs/2011.14670

import copy
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from .dist_loss import cosine_similarity
from .celoss import CELoss
from .triplet import TripletLoss


class IntraDomainScatterLoss(nn.Layer):
    """
    IntraDomainScatterLoss
    
    enhance intra-domain diversity and disarrange inter-domain distributions like confusing multiple styles.

    reference: https://arxiv.org/abs/2011.14670
    """

    def __init__(self, normalize_feature, feature_from):
        super(IntraDomainScatterLoss, self).__init__()
        self.normalize_feature = normalize_feature
        self.feature_from = feature_from

    def forward(self, input, batch):
        domains = batch["domain"]
        inputs = input[self.feature_from]

        if self.normalize_feature:
            inputs = 1. * inputs / (paddle.expand_as(
                paddle.norm(
                    inputs, p=2, axis=-1, keepdim=True), inputs) + 1e-12)

        unique_label = paddle.unique(domains)
        features_per_domain = list()
        for i, x in enumerate(unique_label):
            features_per_domain.append(inputs[x == domains])
        num_domain = len(features_per_domain)
        losses = []
        for i in range(num_domain):
            features_in_same_domain = features_per_domain[i]
            center = paddle.mean(features_in_same_domain, 0)
            cos_sim = cosine_similarity(
                center.unsqueeze(0), features_in_same_domain)
            losses.append(paddle.mean(cos_sim))
        loss = paddle.mean(paddle.stack(losses))
        return {"IntraDomainScatterLoss": loss}


class InterDomainShuffleLoss(nn.Layer):
    """
    InterDomainShuffleLoss

    pull the negative sample of the interdomain and push the negative sample of the intra-domain, 
    so that the inter-domain distributions are shuffled.

    reference: https://arxiv.org/abs/2011.14670
    """

    def __init__(self, normalize_feature=True, feature_from="features"):
        super(InterDomainShuffleLoss, self).__init__()
        self.feature_from = feature_from
        self.normalize_feature = normalize_feature

    def forward(self, input, batch):
        target = batch["label"]
        domains = batch["domain"]
        inputs = input[self.feature_from]

        if self.normalize_feature:
            inputs = 1. * inputs / (paddle.expand_as(
                paddle.norm(
                    inputs, p=2, axis=-1, keepdim=True), inputs) + 1e-12)

        bs = inputs.shape[0]

        # compute distance
        dist_mat = paddle.pow(inputs, 2).sum(axis=1, keepdim=True).expand(
            [bs, bs])
        dist_mat = dist_mat + dist_mat.t()
        dist_mat = paddle.addmm(
            input=dist_mat, x=inputs, y=inputs.t(), alpha=-2.0, beta=1.0)
        dist_mat = paddle.clip(dist_mat, min=1e-12).sqrt()

        is_same_img = np.zeros(shape=[bs, bs], dtype=bool)
        np.fill_diagonal(is_same_img, True)
        is_same_img = paddle.to_tensor(is_same_img)
        is_same_instance = target.reshape([bs, 1]).expand([bs, bs])\
            .equal(target.reshape([bs, 1]).expand([bs, bs]).t())
        is_same_domain = domains.reshape([bs, 1]).expand([bs, bs])\
            .equal(domains.reshape([bs, 1]).expand([bs, bs]).t())

        set_all = []
        set_all.extend([is_same_instance * (is_same_img == False)])
        set_all.extend(
            [(is_same_instance == False) * (is_same_domain == True)])
        set_all.extend([is_same_domain == False])

        is_pos = copy.deepcopy(is_same_img)
        is_neg = paddle.zeros_like(is_same_img, dtype=bool)
        for i, bool_flag in enumerate([0, 0, 1]):
            if bool_flag == 1:
                is_pos = paddle.logical_or(is_pos, set_all[i])

        for i, bool_flag in enumerate([0, 1, 0]):
            if bool_flag == 1:
                is_neg = paddle.logical_or(is_neg, set_all[i])

        dist_ap = list()
        for i in range(dist_mat.shape[0]):
            dist_ap.append(paddle.max(dist_mat[i][is_pos[i]]))
        dist_ap = paddle.stack(dist_ap)

        dist_an = list()
        for i in range(dist_mat.shape[0]):
            dist_an.append(paddle.min(dist_mat[i][is_neg[i]]))
        dist_an = paddle.stack(dist_an)

        y = paddle.ones_like(dist_an)
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        if loss == float('Inf'):
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        return {"InterDomainShuffleLoss": loss}


class CELossForMetaBIN(CELoss):
    def __init__(self, reduction="mean", epsilon=None):
        super().__init__(reduction, epsilon)

    def forward(self, x, batch):
        label = batch["label"]
        return super().forward(x, label)


class TripletLossForMetaBIN(TripletLoss):
    def __init__(self, margin=1, feature_from="feature"):
        super().__init__(margin)
        self.feature_from = feature_from

    def forward(self, input, batch):
        input["feature"] = input[self.feature_from]
        target = batch["label"]
        return super().forward(input, target)
