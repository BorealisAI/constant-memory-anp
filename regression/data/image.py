# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2021, Tung Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TNP (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/tung-nd/TNP-pytorch by Tung Nguyen 
####################################################################################


import torch
from attrdict import AttrDict
from torch.distributions import StudentT

def img_to_task(img, num_ctx=None,
        max_num_points=None, target_all=False, t_noise=None, max_num_target_points=None):

    B, C, H, W = img.shape
    num_pixels = H*W
    img = img.view(B, C, -1)

    if t_noise is not None:
        if t_noise == -1:
            t_noise = 0.09 * torch.rand(img.shape)
        img += t_noise * StudentT(2.1).rsample(img.shape)

    batch = AttrDict()
    max_num_points = max_num_points or num_pixels
    if max_num_target_points is None:
        max_num_target_points = max_num_points
    num_ctx = num_ctx or \
            torch.randint(low=3, high=max_num_points-3, size=[1]).item()
    num_tar = max_num_points - num_ctx if target_all else \
            torch.randint(low=3, high=min(max_num_points-num_ctx, max_num_target_points), size=[1]).item()
    num_points = num_ctx + num_tar
    idxs = torch.cuda.FloatTensor(B, num_pixels).uniform_().argsort(-1)[...,:num_points].to(img.device)
    x1, x2 = idxs//W, idxs%W
    batch.x = torch.stack([
        2*x1.float()/(H-1) - 1,
        2*x2.float()/(W-1) - 1], -1).to(img.device)
    batch.y = (torch.gather(img, -1, idxs.unsqueeze(-2).repeat(1, C, 1))\
            .transpose(-2, -1) - 0.5).to(img.device)

    batch.xc = batch.x[:,:num_ctx]
    batch.xt = batch.x[:,num_ctx:]
    batch.yc = batch.y[:,:num_ctx]
    batch.yt = batch.y[:,num_ctx:]

    return batch
