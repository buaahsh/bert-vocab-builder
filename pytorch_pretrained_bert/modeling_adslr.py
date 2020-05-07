# coding=utf-8
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np
from pathlib import Path
from scipy.stats import truncnorm

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from .file_utils import cached_path
from .config import BertConfig, PREDEFINED_MODEL_CONFIGS, CONFIG_NAME, WEIGHTS_NAME, CONFIG_NAME_ARCHIVE_MAP, PRETRAINED_MODEL_ARCHIVE_MAP

from biunilm.ranking_loss import _list_mle_loss

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    from torch.nn import LayerNorm as BertLayerNorm

logger = logging.getLogger(__name__)


def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def mix_act(x):
    signature = (x >= 0).type_as(x)
    t = 1.12643 * x - math.pi
    small_mask = torch.abs(t) < 1e-5
    pos_mask = t >= 0
    t.masked_fill_(small_mask & pos_mask, 1e-5)
    t.masked_fill_(small_mask & ~pos_mask, -1e-5)
    return signature * (x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))) + (1.0 - signature) * torch.sin(t) / t
    # return signature * gelu(x) + (1.0 - signature) * torch.sin(1.12643 * x - math.pi) / (1.12643 * x - math.pi - 1e-6)


def get_rel_pos_onehot_size(config):
    if config.rel_pos_type == 1:
        rel_pos_onehot_size = 2*config.max_rel_pos+1
    elif config.rel_pos_type == 2:
        rel_pos_onehot_size = config.rel_pos_bins
    if config.seperate_cls_rel_pos:
        rel_pos_onehot_size += 2
    return rel_pos_onehot_size


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu,
          "swish": swish, "mix_act": mix_act}


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size)
        else:
            self.token_type_embeddings = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.prenorm = config.prenorm
        if not self.prenorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = config.hidden_dropout_prob

        if config.emb_scale:
            self.emb_scale = math.sqrt(config.hidden_size)
        else:
            self.emb_scale = None

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        if self.emb_scale is not None:
            words_embeddings *= self.emb_scale
        position_embeddings = self.position_embeddings(position_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings

        if self.token_type_embeddings is not None:
            embeddings = embeddings + \
                self.token_type_embeddings(token_type_ids)

        if not self.prenorm:
            embeddings = self.LayerNorm(embeddings)
        embeddings = F.dropout(embeddings, self.dropout, self.training)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.fast_qkv = config.fast_qkv
        if config.fast_qkv:
            self.qkv_linear = nn.Linear(
                config.hidden_size, 3*self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = config.attention_probs_dropout_prob
        self.rel_pos_type = config.rel_pos_type
        self.untie_rel_pos = config.untie_rel_pos
        self.seperate_cls_rel_pos = config.seperate_cls_rel_pos
        self.rel_pos_onehot_size = get_rel_pos_onehot_size(config)
        if (self.rel_pos_type in (1, 2)) and config.untie_rel_pos:
            self.rel_pos_bias = nn.Linear(
                self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

        self.key_unary = config.key_unary
        if config.key_unary:
            self.key_unary_weight = nn.Parameter(torch.zeros(
                1, self.num_attention_heads, config.hidden_size, 1))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_for_masked(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q += self.q_bias
                v += self.v_bias
            else:
                _sz = (1,) * (q.ndimension()-1) + (-1,)
                q += self.q_bias.view(*_sz)
                v += self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = F.linear(hidden_states, self.key.weight)
            v = self.value(hidden_states)
        return q, k, v

    def forward(self, hidden_states, attention_mask, predict_sequence=None,
                predict_attention_mask=None, history_states=None,
                rel_pos=None, predict_rel_pos=None):
        q, k, v = self.compute_qkv(hidden_states)

        # (B, L, H*D) -> (B, H, L, D)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        query_layer = query_layer / math.sqrt(self.attention_head_size)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        if self.rel_pos_type in (1, 2):
            if self.untie_rel_pos:
                # (B,H,L,L)
                rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
            attention_scores = attention_scores + rel_pos
        if self.key_unary:
            # (B,1,L,I) * (1,H,I,1) -> (B,H,1,L)
            unary_attention_scores = torch.matmul(
                hidden_states.unsqueeze(1), self.key_unary_weight).transpose(-1, -2)
            attention_scores = attention_scores + unary_attention_scores
        attention_scores = attention_scores.float(
        ).masked_fill_(attention_mask, float('-inf'))
        attention_probs = F.softmax(
            attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)

        attention_probs = F.dropout(
            attention_probs, self.dropout, self.training)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if predict_sequence is not None:
            predict_q, predict_k, predict_v = self.compute_qkv(
                predict_sequence)

            predict_query_layer = self.transpose_for_scores_for_masked(
                predict_q)
            predict_key_layer = self.transpose_for_scores_for_masked(
                predict_k)
            predict_value_layer = self.transpose_for_scores_for_masked(
                predict_v)

            num_shuffle = predict_sequence.shape[1]
            extended_key_layer = key_layer.unsqueeze(
                1).expand(-1, num_shuffle, -1, -1, -1)
            merged_key_layer = torch.cat(
                (extended_key_layer, predict_key_layer), dim=-2)
            extended_value_layer = value_layer.unsqueeze(
                1).expand(-1, num_shuffle, -1, -1, -1)
            merged_value_layer = torch.cat(
                (extended_value_layer, predict_value_layer), dim=-2)

            predict_query_layer = predict_query_layer / \
                math.sqrt(self.attention_head_size)
            merged_attention_scores = torch.matmul(
                predict_query_layer, merged_key_layer.transpose(-1, -2))
            if self.rel_pos_type in (1, 2):
                if self.untie_rel_pos:
                    # (B,S,H,P,L+P)
                    predict_rel_pos = self.rel_pos_bias(
                        predict_rel_pos).permute(0, 1, 4, 2, 3)
                merged_attention_scores = merged_attention_scores + predict_rel_pos
            if self.key_unary:
                # (B,L,I), (B,S,P,I) -> (B,S,L+P,I)
                merged_hidden_states = torch.cat((hidden_states.unsqueeze(
                    1).expand(-1, num_shuffle, -1, -1), predict_sequence), dim=-2).unsqueeze(2)
                # (B,S,1,L+P,I) * (1,1,H,I,1) -> (B,S,H,1,L+P)
                unary_attention_scores = torch.matmul(
                    merged_hidden_states, self.key_unary_weight.unsqueeze(1)).transpose(-1, -2)
            merged_attention_scores = merged_attention_scores.float(
            ).masked_fill_(predict_attention_mask, float('-inf'))
            merged_attention_probs = F.softmax(
                merged_attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            merged_attention_probs = F.dropout(
                merged_attention_probs, self.dropout, self.training)

            merged_context_layer = torch.matmul(
                merged_attention_probs, merged_value_layer)
            merged_context_layer = merged_context_layer.permute(
                0, 1, 3, 2, 4).contiguous()
            new_context_layer_shape = merged_context_layer.size()[
                :-2] + (self.all_head_size,)
            merged_context_layer = merged_context_layer.view(
                *new_context_layer_shape)
        else:
            merged_context_layer = None

        return context_layer, merged_context_layer


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.prenorm = config.prenorm
        if self.prenorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, input_tensor, attention_mask,
                predict_hidden_states=None, predict_attention_mask=None, history_states=None,
                rel_pos=None, predict_rel_pos=None):
        if self.prenorm:
            _input_tensor = self.LayerNorm(input_tensor)
            if predict_hidden_states is not None:
                _predict_hidden_states = self.LayerNorm(predict_hidden_states)
        else:
            _input_tensor = input_tensor
            _predict_hidden_states = predict_hidden_states
        self_output, predict_output = self.self(
            _input_tensor, attention_mask,
            predict_sequence=_predict_hidden_states,
            predict_attention_mask=predict_attention_mask,
            history_states=history_states,
            rel_pos=rel_pos, predict_rel_pos=predict_rel_pos)
        attention_output = self.output(self_output, input_tensor)
        if predict_output is not None:
            predict_attention_output = self.output(
                predict_output, predict_hidden_states)
        else:
            predict_attention_output = None
        return attention_output, predict_attention_output


def weight_standarzation(weight, norm=None, eps=1e-6):
    weight_mean = weight.mean(dim=1, keepdim=True)
    weight = weight - weight_mean
    std = torch.sqrt(weight.var(dim=1, keepdim=True) + eps)
    if norm is not None:
        std = std*norm
    weight = weight / std.expand_as(weight)
    return weight


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.ln_after_ffn1 = config.ln_after_ffn1
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.hidden_dropout_prob
        self.prenorm = config.prenorm
        if not self.prenorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.ws_affine = nn.Parameter(torch.ones(config.hidden_size)) if (
            self.ln_after_ffn1 == 13) else None
        self.norm_factor = math.sqrt(config.hidden_size) if (
            self.ln_after_ffn1 in (12, 13)) else None
        self.offline_ws = config.offline_ws

    def forward(self, hidden_states, input_tensor):
        if self.ln_after_ffn1 in (12, 13) and (not self.offline_ws):
            weight = weight_standarzation(self.dense.weight.float(
            ), norm=self.norm_factor)
            weight = weight.type_as(hidden_states)
        else:
            weight = self.dense.weight
        if self.ws_affine is None:
            hidden_states = F.linear(hidden_states, weight, self.dense.bias)
        else:
            hidden_states = self.ws_affine * \
                F.linear(hidden_states, weight) + self.dense.bias
        hidden_states = F.dropout(hidden_states, self.dropout, self.training)
        if self.prenorm:
            return hidden_states + input_tensor
        else:
            return self.LayerNorm(hidden_states + input_tensor)


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.ln_after_ffn1 = config.ln_after_ffn1
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.LayerNorm = BertLayerNorm(
            config.intermediate_size, eps=1e-5) if (config.ln_after_ffn1 not in (0, 8, 12, 13, 14)) else None
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.ws_affine = nn.Parameter(torch.ones(config.intermediate_size)) if (
            self.ln_after_ffn1 in (13, 14)) else None
        self.norm_factor = math.sqrt(config.hidden_size) if (
            self.ln_after_ffn1 in (7, 8, 9, 10, 11, 12, 13, 14)) else None
        self.offline_ws = config.offline_ws

    def forward(self, hidden_states):
        if self.ln_after_ffn1 in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14) and (not self.offline_ws):
            weight = weight_standarzation(self.dense.weight.float(
            ), norm=self.norm_factor)
            weight = weight.type_as(hidden_states)
        else:
            weight = self.dense.weight
        if self.ws_affine is None:
            hidden_states = F.linear(hidden_states, weight, self.dense.bias)
        else:
            hidden_states = self.ws_affine * \
                F.linear(hidden_states, weight) + self.dense.bias

        if self.ln_after_ffn1 in (1, 3, 4, 5, 9, 10):
            hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        if self.ln_after_ffn1 in (2, 6, 7, 11):
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.ln_after_ffn1 = config.ln_after_ffn1
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = config.hidden_dropout_prob
        self.prenorm = config.prenorm
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=1e-5) if (not self.prenorm) else None
        self.ffn2LayerNorm = BertLayerNorm(
            config.hidden_size, eps=1e-5) if (self.ln_after_ffn1 == 5) else None
        self.ws_affine = nn.Parameter(torch.ones(config.hidden_size)) if (
            self.ln_after_ffn1 in (13, 14)) else None
        self.norm_factor = math.sqrt(config.intermediate_size) if (
            self.ln_after_ffn1 in (8, 10, 11, 12, 13, 14)) else None
        self.offline_ws = config.offline_ws

    def forward(self, hidden_states, input_tensor):
        if self.ln_after_ffn1 in (4, 5, 8, 10, 11, 12, 13, 14) and (not self.offline_ws):
            weight = weight_standarzation(self.dense.weight.float(
            ), norm=self.norm_factor)
            weight = weight.type_as(hidden_states)
        else:
            weight = self.dense.weight
        if self.ws_affine is None:
            hidden_states = F.linear(hidden_states, weight, self.dense.bias)
        else:
            hidden_states = self.ws_affine * \
                F.linear(hidden_states, weight) + self.dense.bias
        if self.ln_after_ffn1 == 5:
            hidden_states = self.ffn2LayerNorm(hidden_states)
        hidden_states = F.dropout(hidden_states, self.dropout, self.training)
        if self.prenorm:
            return hidden_states + input_tensor
        else:
            return self.LayerNorm(hidden_states + input_tensor)


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.prenorm = config.prenorm
        if self.prenorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states, attention_mask,
                predict_hidden_states=None, predict_attention_mask=None, history_states=None,
                rel_pos=None, predict_rel_pos=None):
        attention_output, predict_attention_output = self.attention(
            hidden_states, attention_mask,
            predict_hidden_states=predict_hidden_states,
            predict_attention_mask=predict_attention_mask,
            history_states=history_states,
            rel_pos=rel_pos, predict_rel_pos=predict_rel_pos)

        _attention_output = self.LayerNorm(
            attention_output) if self.prenorm else attention_output
        intermediate_output = self.intermediate(_attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        predict_layer_output = None
        if predict_attention_output is not None:
            _predict_attention_output = self.LayerNorm(
                predict_attention_output) if self.prenorm else predict_attention_output
            predict_intermediate_output = self.intermediate(
                _predict_attention_output)
            predict_layer_output = self.output(
                predict_intermediate_output, predict_attention_output)
        return layer_output, predict_layer_output


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    if bidirectional:
        num_buckets //= 2
        # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance /
                                                    max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class CosPositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, all_head_size):
        super(CosPositionalEmbedding, self).__init__()
        self.hidden_size = hidden_size
        freq_seq = torch.arange(
            0, hidden_size, 2, dtype=torch.float)
        inv_freq = 1.0 / (10000 ** (freq_seq / hidden_size))
        self.register_buffer('inv_freq', inv_freq)
        self.cos_pos_linear = nn.Linear(
            hidden_size, all_head_size, bias=False)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        sinusoid_mat = torch.cat(
            [sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return self.cos_pos_linear(sinusoid_mat)


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])
        self.prenorm = config.prenorm
        if self.prenorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.rel_pos_type = config.rel_pos_type
        self.max_rel_pos = config.max_rel_pos
        self.rel_pos_bins = config.rel_pos_bins
        self.untie_rel_pos = config.untie_rel_pos
        self.seperate_cls_rel_pos = config.seperate_cls_rel_pos
        self.rel_pos_onehot_size = get_rel_pos_onehot_size(config)
        if (self.rel_pos_type in (1, 2)) and (not config.untie_rel_pos):
            self.rel_pos_bias = nn.Linear(
                self.rel_pos_onehot_size, config.num_attention_heads, bias=False)

    def convert_rel_pos_mat(self, m):
        if self.rel_pos_type == 1:
            # containing inplace operators
            # (B,S,L+P,L+P)
            merged_rel_pos = m.clamp_(
                min=-self.max_rel_pos, max=self.max_rel_pos).add_(self.max_rel_pos)
        elif self.rel_pos_type == 2:
            merged_rel_pos = relative_position_bucket(
                m, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
        return F.one_hot(merged_rel_pos, num_classes=self.rel_pos_onehot_size)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True,
                predict_hidden_states=None, predict_attention_mask=None,
                prev_embedding=None, prev_encoded_layers=None,
                position_ids=None, predict_token_position_ids=None):
        # history embedding and encoded layer must be simultanously given
        assert (prev_embedding is None) == (prev_encoded_layers is None)

        rel_pos, predict_rel_pos = None, None
        if self.rel_pos_type in (1, 2):
            _L = position_ids.size(-1)
            _P = predict_token_position_ids.size(-1)
            # (B,S,L+P)
            merged_pos_ids = torch.cat((position_ids.unsqueeze(
                1).expand(-1, predict_token_position_ids.size(1), -1), predict_token_position_ids), dim=-1)
            # (B,S,L+P,L+P)
            rel_pos_mat = merged_pos_ids.unsqueeze(-2) - \
                merged_pos_ids.unsqueeze(-1)
            # (B,S,L,L)
            rel_pos = rel_pos_mat[:, 0, :_L, :_L]
            # (B,S,P,L+P)
            predict_rel_pos = rel_pos_mat[:, :, -_P:, :]

            rel_pos = self.convert_rel_pos_mat(rel_pos).type_as(hidden_states)
            if self.seperate_cls_rel_pos:
                rel_pos[:, :, 0, :].fill_(self.rel_pos_onehot_size-2)
                rel_pos[:, :, :, 0].fill_(self.rel_pos_onehot_size-1)
            predict_rel_pos = self.convert_rel_pos_mat(
                predict_rel_pos).type_as(hidden_states)
            if self.seperate_cls_rel_pos:
                predict_rel_pos[:, :, :, 0].fill_(self.rel_pos_onehot_size-1)
            if not self.untie_rel_pos:
                # (B,H,L,L)
                rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
                # (B,S,H,P,L+P)
                predict_rel_pos = self.rel_pos_bias(
                    predict_rel_pos).permute(0, 1, 4, 2, 3)

        all_encoder_layers = []
        assert (prev_embedding is None) and (prev_encoded_layers is None)
        for i, layer_module in enumerate(self.layer):
            hidden_states, predict_hidden_states = layer_module(
                hidden_states, attention_mask,
                predict_hidden_states=predict_hidden_states,
                predict_attention_mask=predict_attention_mask,
                rel_pos=rel_pos, predict_rel_pos=predict_rel_pos)
            if self.prenorm and (i == len(self.layer)-1):
                # pre-layernorm: apply layernorm for the topmost hidden states
                hidden_states = self.LayerNorm(hidden_states)
                predict_hidden_states = self.LayerNorm(predict_hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(
                    (hidden_states, predict_hidden_states))
        if not output_all_encoded_layers:
            all_encoder_layers.append((hidden_states, predict_hidden_states))
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states, bert_model_embedding_weights, task_idx):
        hidden_states = self.transform(hidden_states)
        hidden_states = F.linear(
            hidden_states, bert_model_embedding_weights, self.bias)
        # hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output, bert_model_embedding_weights):
        prediction_scores = self.predictions(
            sequence_output, bert_model_embedding_weights)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        if config.next_sentence_prediction and num_labels > 0:
            self.seq_relationship = nn.Linear(config.hidden_size, num_labels)
        else:
            self.seq_relationship = None

    def forward(self, sequence_output, pooled_output, bert_model_embedding_weights, task_idx):
        prediction_scores = self.predictions(
            sequence_output, bert_model_embedding_weights, task_idx)
        if pooled_output is None or self.seq_relationship is None:
            seq_relationship_score = None
        else:
            seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            # numpy.truncnorm() would take a long time in philly clusters
            # module.weight = torch.nn.Parameter(torch.Tensor(
            #     truncnorm.rvs(-1, 1, size=list(module.weight.data.shape)) * self.config.initializer_range))
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, BertSelfAttention):
            if self.config.att_dnl > 0:
                module.dnl_unary_weight.data.normal_(
                    mean=0.0, std=self.config.initializer_range)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, no_nsp=True, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        tempdir = None
        serialization_dir = None

        def _get_serialization_dir():
            if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
                archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
            else:
                archive_file = pretrained_model_name
            # redirect to the cache, if necessary
            try:
                resolved_archive_file = cached_path(
                    archive_file, cache_dir=cache_dir)
            except FileNotFoundError:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name,
                        ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                        archive_file))
                return None
            if resolved_archive_file == archive_file:
                logger.info("loading archive file {}".format(archive_file))
            else:
                logger.info("loading archive file {} from cache at {}".format(
                    archive_file, resolved_archive_file))
            if os.path.isdir(resolved_archive_file):
                serialization_dir = resolved_archive_file
            else:
                # Extract archive to temp dir
                tempdir = tempfile.mkdtemp()
                logger.info("extracting archive file {} to temp dir {}".format(
                    resolved_archive_file, tempdir))
                with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                    archive.extractall(tempdir)
                serialization_dir = tempdir
            return serialization_dir
        # Load config
        if pretrained_model_name in PREDEFINED_MODEL_CONFIGS:
            # assert state_dict is not None and isinstance(state_dict, dict) and len(state_dict) == 0
            for arg_clean in (
                    'config_path', 'type_vocab_size', 'relax_projection', 'task_idx', 'max_position_embeddings', 'emb_scale', 'ffn_type', 'label_smoothing', 'hidden_dropout_prob', 'attention_probs_dropout_prob', 'reset_dropout_ratio', 'model_act', 'prenorm', 'rel_pos_type', 'max_rel_pos', 'rel_pos_bins', 'seperate_cls_rel_pos', 'untie_rel_pos', 'att_dnl', 'key_unary', 'query_rel_pos', 'fast_qkv', 'ln_after_ffn1', 'offline_ws'):
                if arg_clean in kwargs:
                    del kwargs[arg_clean]
            config = PREDEFINED_MODEL_CONFIGS[pretrained_model_name]
        else:
            if ('config_path' in kwargs) and kwargs['config_path']:
                config_file = kwargs['config_path']
            elif pretrained_model_name in CONFIG_NAME_ARCHIVE_MAP:
                config_file = os.path.join(
                    Path(__file__).parent, CONFIG_NAME_ARCHIVE_MAP[pretrained_model_name])
            else:
                if not serialization_dir:
                    serialization_dir = _get_serialization_dir()
                config_file = os.path.join(serialization_dir, CONFIG_NAME)
            config = BertConfig.from_json_file(config_file)

        # define new type_vocab_size (there might be different numbers of segment ids)
        if 'type_vocab_size' in kwargs:
            config.type_vocab_size = kwargs['type_vocab_size']
        # define new relax_projection
        if ('relax_projection' in kwargs) and kwargs['relax_projection']:
            config.relax_projection = kwargs['relax_projection']
        # define new relax_projection
        if 'task_idx' in kwargs:
            config.task_idx = kwargs['task_idx']
        # define new max position embedding for length expansion
        if ('max_position_embeddings' in kwargs) and kwargs['max_position_embeddings']:
            config.max_position_embeddings = kwargs['max_position_embeddings']
        # embeddings scale by sqrt(dim)
        if 'emb_scale' in kwargs:
            config.emb_scale = kwargs['emb_scale']
        # type of FFN in transformer blocks
        if ('ffn_type' in kwargs) and kwargs['ffn_type']:
            config.ffn_type = kwargs['ffn_type']
        # label smoothing
        if ('label_smoothing' in kwargs) and kwargs['label_smoothing']:
            config.label_smoothing = kwargs['label_smoothing']
        # dropout
        if ('hidden_dropout_prob' in kwargs) and (kwargs['hidden_dropout_prob'] >= 0):
            config.hidden_dropout_prob = kwargs['hidden_dropout_prob']
        if ('attention_probs_dropout_prob' in kwargs) and (kwargs['attention_probs_dropout_prob'] >= 0):
            config.attention_probs_dropout_prob = kwargs['attention_probs_dropout_prob']
        if ('reset_dropout_ratio' in kwargs) and (kwargs['reset_dropout_ratio'] >= 0):
            config.reset_dropout_ratio = kwargs['reset_dropout_ratio']
        # activation function
        if ('model_act' in kwargs) and kwargs['model_act']:
            config.hidden_act = kwargs['model_act']
        # pre-layernorm
        if 'prenorm' in kwargs:
            config.prenorm = kwargs['prenorm']
        # layernorm after ffn1
        if 'ln_after_ffn1' in kwargs:
            config.ln_after_ffn1 = kwargs['ln_after_ffn1']
        if 'offline_ws' in kwargs:
            config.offline_ws = kwargs['offline_ws']
        # relative position
        if 'rel_pos_type' in kwargs:
            config.rel_pos_type = kwargs['rel_pos_type']
        if 'max_rel_pos' in kwargs:
            config.max_rel_pos = kwargs['max_rel_pos']
        if 'rel_pos_bins' in kwargs:
            config.rel_pos_bins = kwargs['rel_pos_bins']
        if 'seperate_cls_rel_pos' in kwargs:
            config.seperate_cls_rel_pos = kwargs['seperate_cls_rel_pos']
        if 'untie_rel_pos' in kwargs:
            config.untie_rel_pos = kwargs['untie_rel_pos']
        # attention scoring function
        if 'att_dnl' in kwargs:
            config.att_dnl = kwargs['att_dnl']
        if 'key_unary' in kwargs:
            config.key_unary = kwargs['key_unary']
        if 'query_rel_pos' in kwargs:
            config.query_rel_pos = kwargs['query_rel_pos']
        # fast QKV computation
        if 'fast_qkv' in kwargs:
            config.fast_qkv = kwargs['fast_qkv']

        if no_nsp:
            config.next_sentence_prediction = False
        logger.info("Model config {}".format(config))

        # clean the arguments in kwargs
        for arg_clean in ('config_path', 'type_vocab_size', 'relax_projection', 'task_idx', 'max_position_embeddings', 'emb_scale', 'ffn_type', 'label_smoothing', 'hidden_dropout_prob', 'attention_probs_dropout_prob', 'reset_dropout_ratio', 'model_act', 'prenorm', 'rel_pos_type', 'max_rel_pos', 'rel_pos_bins', 'seperate_cls_rel_pos', 'untie_rel_pos', 'att_dnl', 'key_unary', 'query_rel_pos', 'fast_qkv', 'ln_after_ffn1', 'offline_ws'):
            if arg_clean in kwargs:
                del kwargs[arg_clean]

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            if not serialization_dir:
                serialization_dir = _get_serialization_dir()
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # initialize new segment embeddings
        _k = 'bert.embeddings.token_type_embeddings.weight'
        if (_k in state_dict) and (config.type_vocab_size != state_dict[_k].shape[0]):
            logger.info("config.type_vocab_size != state_dict[bert.embeddings.token_type_embeddings.weight] ({0} != {1})".format(
                config.type_vocab_size, state_dict[_k].shape[0]))
            if config.type_vocab_size > state_dict[_k].shape[0]:
                # state_dict[_k].data = state_dict[_k].data.resize_(config.type_vocab_size, state_dict[_k].shape[1])
                state_dict[_k].resize_(
                    config.type_vocab_size, state_dict[_k].shape[1])
                if config.type_vocab_size >= 6:
                    # L2R
                    state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
                    # R2L
                    state_dict[_k].data[3, :].copy_(state_dict[_k].data[0, :])
                    # S2S
                    state_dict[_k].data[4, :].copy_(state_dict[_k].data[0, :])
                    state_dict[_k].data[5, :].copy_(state_dict[_k].data[1, :])
            elif config.type_vocab_size < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.type_vocab_size, :]

        # initialize new position embeddings
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict and config.max_position_embeddings != state_dict[_k].shape[0]:
            logger.info("config.max_position_embeddings != state_dict[bert.embeddings.position_embeddings.weight] ({0} - {1})".format(
                config.max_position_embeddings, state_dict[_k].shape[0]))
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                old_size = state_dict[_k].shape[0]
                # state_dict[_k].data = state_dict[_k].data.resize_(config.max_position_embeddings, state_dict[_k].shape[1])
                state_dict[_k].resize_(
                    config.max_position_embeddings, state_dict[_k].shape[1])
                start = old_size
                while start < config.max_position_embeddings:
                    chunk_size = min(
                        old_size, config.max_position_embeddings - start)
                    state_dict[_k].data[start:start+chunk_size,
                                        :].copy_(state_dict[_k].data[:chunk_size, :])
                    start += chunk_size
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.max_position_embeddings, :]

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        model.missing_keys = missing_keys
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            logger.info('\n'.join(error_msgs))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        if config.next_sentence_prediction:
            self.pooler = BertPooler(config)
        else:
            self.pooler = None
        self.apply(self.init_bert_weights)

    def rescale_some_parameters(self):
        def rescale(param, layer_id):
            return torch.nn.Parameter(param / math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.encoder.layer):
            layer.attention.output.dense.weight = rescale(
                layer.attention.output.dense.weight.data, layer_id + 1)
            layer.output.dense.weight = rescale(
                layer.output.dense.weight, layer_id + 1)

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 4:
            extended_attention_mask = attention_mask.unsqueeze(2)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(
        #     dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = (
            1 - extended_attention_mask).type(torch.uint8)
        return extended_attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, position_ids=None,
                predict_sequence_ids=None, predict_token_type_ids=None,
                predict_attention_mask=None, predict_token_position_ids=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        if predict_attention_mask is not None:
            extended_predict_attention_mask = self.get_extended_attention_mask(
                predict_sequence_ids, predict_token_type_ids, predict_attention_mask)
        else:
            extended_predict_attention_mask = None

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids=position_ids)

        if predict_sequence_ids is not None:
            predict_embedding_output = self.embeddings(
                predict_sequence_ids, predict_token_type_ids, position_ids=predict_token_position_ids)
        else:
            predict_embedding_output = None

        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            predict_hidden_states=predict_embedding_output,
            predict_attention_mask=extended_predict_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            position_ids=position_ids, predict_token_position_ids=predict_token_position_ids)
        sequence_output, predict_sequence_output = encoded_layers[-1]
        if self.pooler:
            pooled_output = self.pooler(sequence_output)
        else:
            pooled_output = None
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1][0]
        return encoded_layers, predict_sequence_output, pooled_output


class BertModelIncr(BertModel):
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, output_all_encoded_layers=True, prev_embedding=None,
                prev_encoded_layers=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


class BertForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads."""

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BingRankingScoreHeads(nn.Module):
    def __init__(self, config, num_labels=1):
        super(BingRankingScoreHeads, self).__init__()
        self.rs0 = nn.Linear(config.hidden_size, config.hidden_size)
        self.rs1 = nn.Linear(config.hidden_size, num_labels)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, sequence_output):
        first_token_rep = sequence_output[:,0]
        scores = self.rs1(self.dropout(self.activation(self.rs0(first_token_rep))))
        return scores


class BertForPreTrainingLossMask(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, context_self_encoding=False, rescale_by_sqrt=True, temperature=None):
        super(BertForPreTrainingLossMask, self).__init__(config)
        self.bert = BertModel(config)
        if not config.next_sentence_prediction:
            num_labels = 0
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.apply(self.init_bert_weights)
        if rescale_by_sqrt:
            self.bert.rescale_some_parameters()
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.crit_mask_lm_softmax = nn.Softmax(dim=-1)

        self.num_labels = num_labels
        if config.next_sentence_prediction:
            self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            self.crit_next_sent = None

        self.context_self_encoding = context_self_encoding
        self.vocab_size = config.vocab_size
        self.temperature = temperature
        if self.temperature is not None and temperature > 0:
            self.kl_loss = nn.KLDivLoss(reduction='none')
        else:
            self.kl_loss = None
        # avoid triggering in every iteration
        self.has_set_dropout = None

        self.rank_scorer = BingRankingScoreHeads(config, num_labels=1)

    def set_dropout(self, p):
        if self.has_set_dropout != p:
            self.bert.embeddings.dropout = p
            for i, layer_module in enumerate(self.bert.encoder.layer):
                layer_module.attention.self.dropout = p
                layer_module.attention.output.dropout = p
                layer_module.output.dropout = p
            self.has_set_dropout = p
    
    def flat_tensor(self, input_tensor):
        # size of input tensor: batch_size * list_length * *size
        if input_tensor is None:
            return input_tensor
        old_shape = list(input_tensor.shape)
        if len(old_shape) < 2:
            return input_tensor
        new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
        output_tensor = input_tensor.reshape(new_shape)
        return output_tensor
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None,
                num_tokens_a=None, num_tokens_b=None, mask_order_ids=None, position_ids=None,
                loss_type=0, pseudo_ids=None, pseudo_pos=None, span_ids=None,
                reversed_order=False, require_mlm_loss=False, delta_weight=None, is_mask=None,
                label=None
                ):
        input_shape = list(input_ids.size())
        list_length = input_shape[1]
        real_batch_size = input_shape[0]

        input_ids = self.flat_tensor(input_ids)
        num_tokens_a = self.flat_tensor(num_tokens_a)
        num_tokens_b = self.flat_tensor(num_tokens_b)
        masked_lm_labels = self.flat_tensor(masked_lm_labels)
        masked_pos = self.flat_tensor(masked_pos)
        masked_weights = self.flat_tensor(masked_weights)
        next_sentence_label = self.flat_tensor(next_sentence_label)
        task_idx = self.flat_tensor(task_idx)
        position_ids = self.flat_tensor(position_ids)
        pseudo_ids = self.flat_tensor(pseudo_ids)
        pseudo_pos = self.flat_tensor(pseudo_pos)
        span_ids = self.flat_tensor(span_ids)
        is_mask = self.flat_tensor(is_mask)
        
        predict_attention_mask = None
        predict_token_type_ids = None
        predict_sequence_ids = None
        predict_token_position_ids = None

        if token_type_ids is None and attention_mask is None:
            sequence_length = input_ids.shape[-1]
            batch_size = input_ids.shape[0]

            all_tokens = position_ids >= 0
            base_mask = all_tokens.type_as(input_ids)

            token_type_ids = (position_ids >= num_tokens_a.view(-1, 1)
                              ) & (position_ids < (num_tokens_a + num_tokens_b).view(-1, 1))
            token_type_ids = token_type_ids.type_as(input_ids)

            position_ids = position_ids * base_mask

            # token_type_ids = base_mask - segment_a_mask  # [0] x num_tokens_a + [1] x num_tokens_b + [0] x num_padding

            def get_attention_mask_from_predict_order(order_ids):
                num_ids = order_ids.shape[-1]
                index_matrix = torch.arange(num_ids).view(
                    1, 1, num_ids).to(order_ids.device)
                index_matrix_t = index_matrix.view(1, num_ids, 1)
                self_mask = index_matrix == index_matrix_t

                from_ids = order_ids.view(-1, num_ids, 1)
                to_ids = order_ids.view(-1, 1, num_ids)
                true_tokens = 0 <= to_ids
                true_tokens_mask = (from_ids >= 0) & true_tokens & (
                    to_ids <= from_ids)
                mask_tokens_mask = (
                    from_ids < 0) & true_tokens & (-to_ids > from_ids)
                mask_tokens_mask = mask_tokens_mask | (
                    (from_ids < 0) & (to_ids == from_ids))

                return self_mask | true_tokens_mask | mask_tokens_mask

            if mask_order_ids is not None:
                attention_mask = get_attention_mask_from_predict_order(
                    order_ids=mask_order_ids) & all_tokens.view(-1, 1, sequence_length)
            elif span_ids is not None:
                # span_ids [BatchSize x NumPad]

                num_pred = span_ids.shape[-1]
                if reversed_order:
                    span_ids = torch.cat(
                        (span_ids, num_pred + 2 - span_ids), dim=1)
                    pseudo_ids = torch.cat((pseudo_ids, pseudo_ids), dim=1)

                cuda_shuffle_times = span_ids.shape[1]
                span_ids = span_ids.view(
                    batch_size, cuda_shuffle_times, num_pred)
                # predict_order = torch.gather(input=predict_weight, dim=-1, index=span_ids).view(-1, num_pred)
                predict_order = torch.cat((span_ids, -span_ids), dim=-1)
                # ground true tokens, mask tokens
                predict_attention_mask = get_attention_mask_from_predict_order(
                    order_ids=predict_order).view(batch_size, cuda_shuffle_times, num_pred * 2, num_pred * 2)
                predict_base_mask = torch.cat(
                    (masked_weights, masked_weights), dim=-1).type_as(predict_attention_mask)
                predict_attention_mask = predict_attention_mask & predict_base_mask.view(
                    batch_size, 1, 1, num_pred * 2)
                predict_token_type_ids = (pseudo_pos >= num_tokens_a.view(-1, 1)) & (
                    pseudo_pos < (num_tokens_a + num_tokens_b).view(-1, 1))
                predict_token_type_ids = predict_token_type_ids.type_as(
                    masked_pos)
                predict_attention_mask = predict_attention_mask.to(input_ids)

                if is_mask is not None:
                    basic_mask = all_tokens.view(batch_size, 1, 1, sequence_length) & \
                        (is_mask == 0).view(batch_size, 1, 1, sequence_length)
                    basic_mask = basic_mask.type_as(input_ids)
                else:
                    basic_mask = base_mask.view(
                        batch_size, 1, 1, sequence_length)

                context_part = basic_mask.expand(-1,
                                                 cuda_shuffle_times, num_pred * 2, -1)
                predict_attention_mask = torch.cat(
                    (context_part, predict_attention_mask), dim=-1)

                attention_mask = base_mask

                masked_tokens = pseudo_ids.view(
                    batch_size, cuda_shuffle_times, num_pred)

                # predict_token_type_ids [BatchSize x 1 x 2NumPad]
                predict_token_type_ids = torch.cat(
                    (predict_token_type_ids, predict_token_type_ids), dim=-1).view(batch_size, 1, num_pred * 2)
                # predict_token_position_ids [BatchSize x 1 x 2NumPad]
                predict_token_position_ids = torch.cat(
                    (pseudo_pos, pseudo_pos), dim=-1).view(batch_size, 1, num_pred * 2)
                # predict_sequence_ids [BatchSize x CST x 2NumPad]
                predict_sequence_ids = torch.cat((masked_lm_labels.view(
                    batch_size, 1, num_pred).expand(-1, cuda_shuffle_times, -1), masked_tokens), dim=-1)
            else:
                attention_mask = base_mask

            attention_mask = attention_mask.type_as(input_ids)

        sequence_output, predict_sequence_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, position_ids=position_ids,
            predict_sequence_ids=predict_sequence_ids, predict_token_type_ids=predict_token_type_ids,
            predict_attention_mask=predict_attention_mask, predict_token_position_ids=predict_token_position_ids,
            output_all_encoded_layers=False)

        if masked_lm_labels is None or next_sentence_label is None:
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output, self.bert.embeddings.word_embeddings.weight, task_idx=task_idx)
            return prediction_scores, seq_relationship_score

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        # masked lm
        masked_lm_probs = None
        if predict_sequence_output is None or require_mlm_loss:
            sequence_output_masked = gather_seq_out_by_pos(
                sequence_output, masked_pos)
            prediction_scores_masked, seq_relationship_score = self.cls(
                sequence_output_masked, pooled_output, self.bert.embeddings.word_embeddings.weight, task_idx=task_idx)
            masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
            masked_lm_loss_mean = loss_mask_and_normalize(
                masked_lm_loss.float(), masked_weights)
            if delta_weight is not None:
                masked_lm_probs = (-masked_lm_loss.float()).exp()
        else:
            masked_lm_loss_mean = None
            seq_relationship_score = None

        delta_loss = None
        teacher_loss = None

        if predict_sequence_output is not None:
            sequence_output_masked = predict_sequence_output[:,
                                                             :, num_pred:, :]
            prediction_scores_masked, seq_relationship_score = self.cls(
                sequence_output_masked, pooled_output, self.bert.embeddings.word_embeddings.weight, task_idx=task_idx)
            # prediction_scores_masked = [BatchSize, CST, NumPred, VocabSize]

            masked_lm_labels = masked_lm_labels.unsqueeze(
                1).expand(-1, cuda_shuffle_times, -1).contiguous().view(-1, num_pred)

            if loss_type == 0:
                masked_weights = masked_weights.unsqueeze(
                    1).expand(-1, cuda_shuffle_times, -1).contiguous().view(-1, num_pred)
            prediction_scores_masked = prediction_scores_masked.view(
                -1, num_pred, self.vocab_size)

            predict_masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)

            if loss_type > 0:
                predict_masked_lm_loss = predict_masked_lm_loss.view(
                    batch_size, cuda_shuffle_times, num_pred).contiguous()
                if loss_type == 1:
                    predict_masked_lm_loss, _ = predict_masked_lm_loss.min(1)
                elif loss_type == 2:
                    predict_masked_lm_loss, _ = predict_masked_lm_loss.max(1)

            pseudo_mlm_loss = loss_mask_and_normalize(
                predict_masked_lm_loss.float(), masked_weights)

            if delta_weight is not None:
                predict_masked_lm_probs = (
                    -predict_masked_lm_loss.float()).exp()
                delta = torch.clamp_min(
                    masked_lm_probs - predict_masked_lm_probs, 0.0)
                delta_loss = loss_mask_and_normalize(delta, masked_weights)

            if self.kl_loss is not None:
                teacher_tokens_masked = predict_sequence_output[:,
                                                                :, :num_pred, :]
                teacher_scores_masked, _ = self.cls(
                    teacher_tokens_masked, pooled_output, self.bert.embeddings.word_embeddings.weight, task_idx=task_idx)

                teacher_scores_masked = teacher_scores_masked.view(
                    -1, num_pred, self.vocab_size)

                teacher_probs = F.softmax(
                    teacher_scores_masked.float() / self.temperature, dim=-1).detach()
                predict_probs = F.softmax(
                    prediction_scores_masked.float() / self.temperature, dim=-1)

                kl_loss = self.kl_loss(predict_probs.permute(
                    2, 0, 1).log(), teacher_probs.permute(2, 0, 1)).sum(0)
                # kl_loss = teacher_probs * (teacher_probs / predict_probs).log()
                # self.kl_loss(teacher_log_probs.permute(2, 0, 1), predict_probs.permute(2, 0, 1)).sum(0)
                teacher_loss = loss_mask_and_normalize(kl_loss, masked_weights)
        else:
            pseudo_mlm_loss = None
            seq_relationship_score = None

        # next sentence
        if self.crit_next_sent:
            next_sentence_loss = self.crit_next_sent(
                seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))
        else:
            next_sentence_loss = None

        # ranking loss
        if label is not None:
            rank_score = self.rank_scorer(sequence_output)
            rank_score = rank_score.reshape([real_batch_size, list_length])
            rank_weights = (task_idx == 0).to(torch.float).reshape([real_batch_size])

            rank_label = label
            rank_loss = _list_mle_loss(labels=rank_label, logits=rank_score, weights=rank_weights) * 0.1
        else:
            rank_loss = None

        return masked_lm_loss_mean, pseudo_mlm_loss, next_sentence_loss, delta_loss, teacher_loss, rank_loss
