# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.
Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class ConvCompress(nn.Module):
    def __init__(self, d_model, ratio = 4, groups = 1):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, ratio, stride = ratio, groups = groups)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)

class CustomizedAttn(nn.Module):
    def __init__(self, d_model=24, nhead=2,  compression_factor = 2, dropout=0.1):
        super().__init__()
        assert (d_model % nhead) == 0, 'dimension must be divisible by number of heads'
        self.heads = nhead
        self.d_model=d_model
        self.compression_factor = compression_factor
        self.compress_fn = ConvCompress(d_model, compression_factor, groups = nhead)

        self.to_k = nn.Linear(d_model,d_model, bias = False)
        self.to_q = nn.Linear(d_model,d_model, bias = False)
        self.to_v=nn.Linear(d_model, d_model, bias = False)
        self.to_out = nn.Linear(d_model, d_model)
    #    self.layernorm = nn.BatchNorm1d(d_model, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout = nn.Dropout(dropout)
        #to modify the first dimension according to the batch number
        self.null_k = nn.Parameter(torch.zeros(10, 1, d_model))
        self.null_v = nn.Parameter(torch.zeros(10, 1, d_model))
      
    def forward(self, k, q, value):   
        k=k.transpose(0,1)
        q=q.transpose(0,1)  
        value=value.transpose(0,1)
        assert self.d_model == k.shape[2], "embed_dim must be equal to the number of 3rd dimension"
        b, t, d, h, cf = *k.shape, self.heads, self.compression_factor
        k= self.to_k(k)
        q= self.to_q(q)
        v= self.to_v(value)
        padding = cf - (t % cf)
        if padding != 0:
            k, v = map(lambda t: F.pad(t, (0, 0, padding, 0)), (k, v))
        k, v = map(self.compress_fn, (k, v))
        # attach a null key and value, in the case that the first query has no keys to pay attention to
        k = torch.cat((self.null_k, k), dim=1)
        v = torch.cat((self.null_v, v), dim=1)
        q, k, v = map(lambda t: t.reshape(*t.shape[:2], self.heads, -1).transpose(1, 2), (q, k, v))
        # attention
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * d ** -0.5
        attn = dots.softmax(dim=-1)
        # dropout
    #    attn=self.layernorm(attn)
        attn = self.dropout(dots)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # split heads and combine
        out = out.transpose(1, 2).reshape(b, t, d)
        out=self.to_out(out)
        return out.transpose(0,1)
      
      
class Transformer(nn.Module):

    def __init__(self, d_model=64, nhead=8, num_encoder_layers=2, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src, pos=pos_embed)

        return memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = CustomizedAttn(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

      #  src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
           #                   key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(q,k, value=src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(k, q,value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
