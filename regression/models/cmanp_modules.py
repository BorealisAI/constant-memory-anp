# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2021, Phil Wang
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Perceiver (https://arxiv.org/abs/2103.03206) implementation
# from https://github.com/lucidrains/Perceiver-pytorch by Phil Wang
####################################################################################

import torch
from torch import nn, einsum
from einops import rearrange, repeat

from models.lbanp_modules import _get_clones, default, exists, FeedForward


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(
            context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        if kwargs.get('return_state', False):
            ret, state = self.fn(x, **kwargs)
            return ret + x, state
        else:
            ret = self.fn(x, **kwargs)
            return ret + x

    def update(self, query_latents, new_ctx_encoding, state):
        x = self.norm(query_latents)

        if exists(self.norm_context):
            normed_new_ctx_encoding = self.norm_context(new_ctx_encoding)
            out, state = self.fn.update(x, normed_new_ctx_encoding, state)
        else:
            out, state = self.fn.update(x, new_ctx_encoding, state)
        return out + x, state


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.softplus = nn.Softplus(beta=1, threshold=20)

    def forward(self, x, context=None, mask=None, return_state=False):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        qu, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', qu, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        if return_state:
            constant = torch.logsumexp(sim, dim=-1).unsqueeze(-1)
            attn = torch.exp(sim - constant)  # Essentially a softmax
        else:
            attn = sim.softmax(dim=-1)

        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out2 = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        if return_state:
            state = {
                'constant': constant,
                'embedding': out.clone(),
            }
            return self.to_out(out2), state
        else:
            return self.to_out(out2)

    def update(self, query_latents, new_ctx_encoding, state):
        if len(query_latents.shape) == 2 and len(new_ctx_encoding.shape) == 3:
            # Ensures the query latents are the correct size
            query_latents = query_latents.unsqueeze(0)

        h = self.heads
        qu = self.to_q(query_latents)
        k, v = self.to_kv(new_ctx_encoding).chunk(2, dim=-1)
        qu, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (qu, k, v))
        sim = einsum('b i d, b j d -> b i j', qu, k) * self.scale

        old_constant = state['constant']
        log_partial_diff = torch.logsumexp(
            (sim - old_constant), dim=-1, keepdim=True)

        constant_diff = self.softplus(log_partial_diff)
        new_constant = old_constant + constant_diff  # Update constant!
        new_ctx_embedding = v

        update_embedding_part = einsum(
            'b i j, b j d -> b i d', torch.exp(sim - new_constant), new_ctx_embedding)
        out = torch.exp(old_constant - new_constant) * \
            state['embedding'] + update_embedding_part

        new_state = {
            'constant': new_constant,
            'embedding': out.clone()
        }

        out2 = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out2), new_state


class CMAB(nn.Module):
    """
        Constant Memory Attention Block (CMAB)
    """

    def __init__(self,
                 num_latents: int,
                 d_model: int, latent_dim: int, nhead: int, dim_feedforward: int, dropout: float,
                 CMAB_block_size: int,
                 norm_first: bool,
                 ):
        super(CMAB, self).__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.latents = nn.Parameter(torch.randn(
            num_latents, self.latent_dim), requires_grad=True)  # Learnable latents

        if CMAB_block_size is not None:
            self.CMAB_block_size = CMAB_block_size
        else:
            self.CMAB_block_size = int(1e9)

        assert (self.latent_dim % nhead == 0)
        assert norm_first

        self.layer_latent_self_attn = PreNorm(self.latent_dim, Attention(
            self.latent_dim, heads=nhead, dim_head=self.latent_dim // nhead, dropout=dropout))
        self.layer_latent_ff = PreNorm(self.latent_dim, FeedForward(
            self.latent_dim, dim_feedforward=dim_feedforward, dropout=dropout))
        self.layer_cross_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, context_dim=self.d_model,
                                        heads=nhead, dim_head=self.latent_dim // nhead, dropout=dropout), context_dim=self.d_model)
        self.layer_cross_ff = PreNorm(self.latent_dim, FeedForward(
            self.latent_dim, dim_feedforward=dim_feedforward, dropout=dropout))

        self.latent_self_attn = PreNorm(self.latent_dim, Attention(
            self.latent_dim, heads=nhead, dim_head=self.latent_dim // nhead, dropout=dropout))
        self.latent_ff = PreNorm(self.latent_dim, FeedForward(
            self.latent_dim, dim_feedforward=dim_feedforward, dropout=dropout))
        self.cross_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, context_dim=self.latent_dim,
                                  heads=nhead, dim_head=self.latent_dim // nhead, dropout=dropout), context_dim=latent_dim)
        self.cross_ff = PreNorm(self.latent_dim, FeedForward(
            self.latent_dim, dim_feedforward=dim_feedforward, dropout=dropout))

    def compute_ctx_latent_encodings(self, context_encodings):
        b, num_context, *axis = context_encodings.shape
        layer_latents = repeat(self.latents, 'n d -> b n d', b=b)

        if self.training:
            layer_ctx_encoding, state = self.layer_cross_attn(
                layer_latents, context=context_encodings, return_state=True)
        else:
            for step in range(0, num_context, self.CMAB_block_size):
                num_context_batch = min(
                    step+self.CMAB_block_size, num_context) - step
                block_encs = context_encodings[:, step:step+num_context_batch]
                if step == 0:
                    layer_ctx_encoding, state = self.layer_cross_attn(
                        layer_latents, context=block_encs, return_state=True)
                else:
                    layer_ctx_encoding, state = self.layer_cross_attn.update(
                        layer_latents, new_ctx_encoding=block_encs, state=state)
        return layer_ctx_encoding, state

    def update(self, new_ctx_encoding, latents, state):
        b, *axis = new_ctx_encoding.shape
        layer_latents = repeat(self.latents, 'n d -> b n d', b=b)

        layer_cross_attn_encoding, state = self.layer_cross_attn.update(
            layer_latents, new_ctx_encoding, state)
        ret = self.forward_without_explicit_ctx(
            latents, layer_cross_attn_encoding)

        return ret, state

    def forward_without_explicit_ctx(self, latents, layer_ctx_encoding):
        # forward function of CMAB, skipping the first cross attention -- instead using its output "layer_ctx_encoding"
        layer_ctx_encoding = self.layer_cross_ff(
            layer_ctx_encoding)  # First Cross Attention

        layer_ctx_encoding = self.layer_latent_self_attn(
            layer_ctx_encoding)  # First Self Attention
        layer_ctx_encoding = self.layer_latent_ff(layer_ctx_encoding)

        x = latents
        # Second Cross Attention
        x = self.cross_attn(x, context=layer_ctx_encoding)
        x = self.cross_ff(x)

        x = self.latent_self_attn(x)  # Second Self Attention
        x = self.latent_ff(x)
        return x

    def forward(self, context_encodings, latents):
        layer_ctx_encoding, state = self.compute_ctx_latent_encodings(
            context_encodings)
        return self.forward_without_explicit_ctx(latents, layer_ctx_encoding=layer_ctx_encoding), state


class CMANPEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(CMANPEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def update(self, new_ctx_encoding, latents, states):
        # Incorporate state. Include a state for each layer I think?
        layer_outputs = []
        new_states = []
        for layer, state in zip(self.layers, states):
            latents, state = layer.update(new_ctx_encoding, latents, state)
            layer_outputs.append(latents)
            new_states.append(state)
        return layer_outputs, new_states  # Return updated layer outputs

    def forward(self, context_encodings, latents):
        b, *axis = context_encodings.shape
        latents = repeat(latents, 'n d -> b n d', b=b)

        layer_outputs = []
        layer_states = []
        for layer in self.layers:
            latents, state = layer(context_encodings, latents)
            layer_outputs.append(latents)
            layer_states.append(state)
        return layer_outputs, layer_states
