# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2022, Tung Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# decode function is based on the TNP-ND (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/tung-nd/TNP-pytorch by Tung Nguyen
####################################################################################

import torch
import torch.nn as nn
from attrdict import AttrDict

from models.modules import build_mlp
from models.cmanp import CMANP


class CMANPAND(CMANP):
    def __init__(
        self,
        num_latents,
        num_latents_per_layer,
        AND_block_size,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        CMAB_block_size,
        num_std_layers,
        norm_first=True,
        bound_std=False,
        cov_approx='cholesky',
        prj_dim=5,
        prj_depth=4,
        diag_depth=4
    ):
        super(CMANPAND, self).__init__(
            num_latents,
            num_latents_per_layer,
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            CMAB_block_size,
            norm_first,
            bound_std
        )

        # AND Specific arg
        if AND_block_size is not None:
            self.AND_block_size = AND_block_size
        else:
            self.AND_block_size = int(1e9)

        # NotDiagonal Components -- Originally from TNPND
        assert cov_approx in ['cholesky', 'lowrank']
        self.cov_approx = cov_approx

        # Mean, Std Networks
        self.mean_net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y)
        )

        # Encoder and Projector Network -- only applied to blocks at a time at inference
        std_encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.std_encoder = nn.TransformerEncoder(
            std_encoder_layer, num_std_layers)
        self.projector = build_mlp(
            d_model, dim_feedforward, prj_dim*dim_y, prj_depth)

        if cov_approx == 'lowrank':
            self.diag_net = build_mlp(
                d_model, dim_feedforward, dim_y, diag_depth)

    # NotDiagonal (ND) Function
    def decode(self, out_encoder, batch_size, dim_y, num_target):
        mean = self.mean_net(out_encoder).view(batch_size, -1)

        out_std_encoder = self.std_encoder(out_encoder)
        std_prj = self.projector(out_std_encoder)
        std_prj = std_prj.view((batch_size, num_target*dim_y, -1))
        if self.cov_approx == 'cholesky':
            std_tril = torch.bmm(std_prj, std_prj.transpose(1, 2))
            std_tril = std_tril.tril()
            if self.bound_std:
                diag_ids = torch.arange(num_target*dim_y, device='cuda')
                std_tril[:, diag_ids, diag_ids] = 0.05 + 0.95 * \
                    torch.tanh(std_tril[:, diag_ids, diag_ids])
            pred_tar = torch.distributions.multivariate_normal.MultivariateNormal(
                mean, scale_tril=std_tril)
        else:
            diagonal = torch.exp(self.diag_net(out_encoder)
                                 ).view((batch_size, -1, 1))
            std = torch.bmm(std_prj, std_prj.transpose(1, 2)) + \
                torch.diag_embed(diagonal.squeeze(-1))
            pred_tar = torch.distributions.multivariate_normal.MultivariateNormal(
                mean, covariance_matrix=std)

        return pred_tar

    # AutoregressiveNotDiagonal (AND) Components

    def forward(self, batch):
        if self.training:
            outs = self.forward_train(batch)
        else:
            outs = self.forward_eval(batch)

        outs.tar_ll = outs.tar_ll.mean()
        outs.loss = - (outs.tar_ll)
        return outs

    def forward_train(self, batch):
        # Training is same as Not-Diagonal variants
        batch_size = batch.x.shape[0]
        dim_y = batch.y.shape[-1]
        num_target = batch.xt.shape[1]

        out_encoder = self.get_predict_encoding(batch)
        pred_tar = self.decode(out_encoder, batch_size, dim_y, num_target)
        outs = AttrDict()
        outs.tar_ll = pred_tar.log_prob(batch.yt.reshape(batch_size, -1))
        outs.mean_std = torch.mean(pred_tar.covariance_matrix)
        return outs

    def forward_eval(self, batch):
        # At evaluation, the autoregressive component is used. Evaluation is similar to that of TNP-A
        xc, yc, xt, yt = batch.xc, batch.yc, batch.xt, batch.yt
        batch_size, num_target, dim_y = xc.shape[0], xt.shape[1], yc.shape[-1]
        batch_stacked = AttrDict()
        batch_stacked.xc, batch_stacked.yc = xc, yc

        context_encodings, state = self.get_context_encoding(
            batch_stacked, return_state=True)

        tar_ll = []
        for step in range(0, num_target, self.AND_block_size):
            num_target_batch = min(step+self.AND_block_size, num_target) - step
            batch_stacked.xt = xt[:, step:step+num_target_batch]
            out_encoder = self.get_predict_encoding(
                batch_stacked, context_encodings=context_encodings)
            pred_tar = self.decode(
                out_encoder, batch_size, dim_y, num_target_batch)

            block_yt = yt[:, step:step+num_target_batch].flatten(-2, -1)
            tar_ll.append(pred_tar.log_prob(block_yt))
            batch_stacked.yt = yt[:, step:step+num_target_batch]

            if step + num_target_batch != num_target:
                context_encodings, state = self.update(
                    batch_stacked.xt, batch_stacked.yt, state)

        outs = AttrDict()
        # Average across number of samples
        outs.tar_ll = sum(tar_ll) / num_target
        outs.mean_std = torch.mean(pred_tar.covariance_matrix)
        return outs
