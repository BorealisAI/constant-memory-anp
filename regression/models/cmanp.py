# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.cmanp_modules import CMAB, CMANPEncoder
from models.lbanp import LBANP


class CMANP(LBANP):
    """
        Constant Memory Attentive Neural Process (LBANPs)
    """

    def __init__(
        self,
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
        norm_first=True,
        bound_std=False
    ):
        super().__init__(
            num_latents,
            dim_x,
            dim_y,
            d_model,
            emb_depth,
            dim_feedforward,
            nhead,
            dropout,
            num_layers,
            norm_first,
            bound_std
        )

        # Context Related:
        encoder_layer = CMAB(num_latents_per_layer, d_model, self.latent_dim,
                             nhead, dim_feedforward, dropout, CMAB_block_size, norm_first)
        self.encoder = CMANPEncoder(encoder_layer, num_layers)

    def get_context_encoding(self, batch, return_state=False):
        # Perform Encoding
        x_y_ctx = torch.cat((batch.xc, batch.yc), dim=-1)
        context_embeddings = self.embedder(x_y_ctx)
        context_encodings, state = self.encoder(
            context_embeddings, self.latents)

        if return_state:
            return context_encodings, state
        else:
            return context_encodings

    def update(self, x, y, state):
        # Efficient update mechanism using CMAB's efficient update property
        new_x_y_ctx = torch.cat((x, y), dim=-1)
        new_ctx_encoding = self.embedder(new_x_y_ctx)
        latents = torch.stack([self.latents]*new_ctx_encoding.shape[0], dim=0)
        context_encodings, new_state = self.encoder.update(
            new_ctx_encoding, latents, state)
        return context_encodings, new_state
