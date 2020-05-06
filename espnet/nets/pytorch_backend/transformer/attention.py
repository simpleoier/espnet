import math

import numpy
import torch
from torch import nn
import torch.nn.functional as F


def multi_headed_attention(n_head, n_feat, dropout_rate, attention_type='self_attn',
                           max_span=100, span_ramp=2, span_init=0, span_ratio=0.5, ratio_adaptive=False,
                           causal_flag=False):
    if attention_type == 'self_attn':
        return MultiHeadedAttention(n_head, n_feat, dropout_rate)
    elif attention_type == 'self_attn_adaptive_span':
        return MultiHeadedAttentionAdaptiveSpan(n_head, n_feat, dropout_rate,
                                                max_span=max_span, span_ramp=2, span_init=span_init,
                                                span_ratio=span_ratio)
    elif attention_type == 'self_attn_dynamic_span':
        return MultiHeadedAttentionDynamicSpan(n_head, n_feat, dropout_rate,
                                               max_span=max_span, span_ramp=2, span_init=span_init,
                                               span_ratio=span_ratio)
    elif attention_type == 'self_attn_fixed_span':
        return MultiHeadedAttentionFixedSpan(n_head, n_feat, dropout_rate,
                                             span_size=max_span, span_ratio=span_ratio)
    elif attention_type in ['self_attn2', 'self_attn_fixed_span2', 'self_attn_adaptive_span2', 'self_attn_dynamic_span2']:
        span_types = dict(
            self_attn2=None,
            self_attn_fixed_span2='fixed',
            self_attn_adaptive_span2='adaptive',
            self_attn_dynamic_span2='dynamic',
        )
        return MultiHeadedAttention2(n_head, n_feat, dropout_rate, span_type=span_types[attention_type],
                                     max_span=max_span, span_ramp=span_ramp, span_init=span_init, span_ratio=span_ratio, ratio_adaptive=ratio_adaptive, causal_flag=causal_flag)
    else:
        raise NotImplementedError(attention_type)


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


class MultiHeadedAttentionTimeRestricted(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttentionTimeRestricted, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time1, time2, size)
        :param torch.Tensor value: (batch, time1, time2, size)
        :param torch.Tensor mask: (batch, 1, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        window = key.size(-2)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k).unsqueeze(3)  # (batch, time1, head, 1, d_k)
        k = self.linear_k(key).view(n_batch, -1, window, self.h, self.d_k)  # (batch, time1, window, head, d_k)
        v = self.linear_v(value).view(n_batch, -1, window, self.h, self.d_k)  # (batch, time1, window, head, d_k)
        q = q.permute(0, 2, 1, 3, 4)  # (batch, head, time1, 1, d_k)
        k = k.permute(0, 3, 1, 2, 4)  # (batch, head, time1, window, d_k)
        v = v.permute(0, 3, 1, 2, 4)  # (batch, head, time1, window, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, 1, window)
        scores = scores.squeeze(3)  # (batch, head, time1, window)
        if mask is not None:
            #mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            mask = mask.eq(0)  # (batch, time1, 1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, window)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, window)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn.unsqueeze(-2), v).squeeze(-2)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)


def unfold_mask(mask, pad_size, dtype=torch.uint8):
    """Take the diagonal window of a mask, as a unfold operation."""
    b, t1, t2 = mask.size()
    span = pad_size[0] + pad_size[1] + 1
    mask = F.pad(mask, pad_size, 'constant', 0)  # (batch, time1, time2+span-1)
    select_mask = torch.ones(t1, t2+span-1, device=mask.device, dtype=dtype)  # (time1, time2+span-1)
    select_mask = torch.eq(select_mask.tril(-1), select_mask.triu(span)).unsqueeze(0)  # (1, time1, time2+span-1)
    return mask.masked_select(select_mask.to(mask.device)).view(b, t1, span)  # (batch, time1, span)


class MultiHeadedAttentionAdaptiveSpan(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """
    def __init__(self, n_head, n_feat, dropout_rate, max_span=100, span_ramp=2, span_init=0, span_ratio=0.5):
        super(MultiHeadedAttentionAdaptiveSpan, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.max_span = max_span
        self.span_mask = AdaptiveSpanMask(n_feat=n_feat,
                                          n_head=n_head,
                                          max_span=self.max_span,
                                          span_ramp=span_ramp,
                                          init_val=span_init,
                                          shape=(n_head, 1, 1))
        self.span_ratio = span_ratio


    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        # For self-attention only, time1 == time2
        span_size = self.span_mask.get_current_max_size()
        pad_size = (int(span_size*self.span_ratio), span_size-int(span_size*self.span_ratio)-1)
        k = F.pad(k, (0, 0, *pad_size), 'constant', 0).unfold(-2, span_size, 1)  # (batch, head, time2, d_k, span)
        v = F.pad(v, (0, 0, *pad_size), 'constant', 0).unfold(-2, span_size, 1)  # (batch, head, time2, d_k, span)

        # (batch, head, time1, 1, d_k) x (batch, head, time2, d_k, span) = (batch, head, time1, 1, span)
        scores = torch.matmul(q.unsqueeze(-2), k).squeeze(-2) / math.sqrt(self.d_k)  # (batch, head, time1, span)
        if mask is not None:
            if mask.size(1) == 1:
                mask = mask.expand(-1, query.size(1), -1)  # (batch, time1, time2)
            mask = unfold_mask(mask, pad_size).unsqueeze(1).eq(0)  # (batch, 1, time1, span)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, span)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        self.attn = self.span_mask(self.attn)
        self.attn = self.attn / (self.attn.sum(-1, keepdim=True) + 1e-8)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn.unsqueeze(-2), v.transpose(-2, -1)).squeeze(-2)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def get_mean_span(self):
        """a loss term for regularizing the span length"""
        return self.span_mask.span_size.mean() * self.max_span

    def clamp_param(self):
        self.span_mask.clamp_param()


class AdaptiveSpanMask(nn.Module):
    """Soft masking function for adaptive size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.
    Args:
        max_span: maximum size (i.e. input dimension)
        span_ramp: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """
    def __init__(self, n_feat, n_head, max_span, span_ramp=2, init_val=1, shape=(1,)):
        super(AdaptiveSpanMask, self).__init__()
        self.max_span = max_span
        self.span_ramp = span_ramp
        self.span_size = nn.Parameter(torch.zeros(*shape) + init_val)
        mask_template = torch.linspace(1 - max_span, 0, steps=max_span)
        self.register_buffer('mask_template', mask_template)

    def forward(self, x):
        mask = self.mask_template + self.span_size * self.max_span
        mask = mask / self.span_ramp + 1
        mask = mask.clamp(0, 1)
        if x.size(-1) < self.max_span:
            mask = mask[:, :, -x.size(-1):]
        x = x * mask
        return x

    def get_current_max_size(self, include_ramp=True):
        current_size = math.ceil(self.span_size.max().item() * self.max_span)
        if include_ramp:
            current_size += self.span_ramp
        current_size = max(0, min(self.max_span, current_size))
        return current_size

    def get_current_avg_size(self, include_ramp=True):
        current_size = math.ceil(self.span_size.mean().item() * self.max_span)
        if include_ramp:
            current_size += self.span_ramp
        current_size = max(0, min(self.max_span, current_size))
        return current_size

    def clamp_param(self):
        self.span_size.data.clamp_(0.3, 1)


class MultiHeadedAttentionDynamicSpan(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """
    def __init__(self, n_head, n_feat, dropout_rate, max_span=50, span_ramp=2, span_init=0, span_ratio=0.5):
        super(MultiHeadedAttentionDynamicSpan, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.max_span = max_span
        self.span_mask = DynamicSpanMask(n_feat=n_feat,
                                         n_head=n_head,
                                         max_span=self.max_span,
                                         span_ramp=span_ramp,
                                         init_val=span_init)
        self.span_ratio = span_ratio


    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, 1, time1) or (batch, time1, time1) in decoder self-attention.
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query)
        # Dynamic span
        max_window_size = self.span_mask.get_current_max_size(q)
        q = q.view(n_batch, -1, self.h, self.d_k)  # (batch, time1, head, d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)  # (batch, time2, head, d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        pad_size = (int(max_window_size*self.span_ratio), max_window_size-int(max_window_size*self.span_ratio)-1)
        k = F.pad(k, (0, 0, *pad_size), "constant", 0).unfold(-2, max_window_size, 1)  # (batch, head, time2, d_k, window)
        v = F.pad(v, (0, 0, *pad_size), "constant", 0).unfold(-2, max_window_size, 1)  # (batch, head, time2, d_k, window)

        # (batch, head, time1, 1, d_k) x (batch, head, time2, d_k, window) = (batch, head, time1, 1, window)
        scores = torch.matmul(q.unsqueeze(-2), k).squeeze(-2) / math.sqrt(self.d_k)  # (batch, head, time1, max_window)
        if mask is not None:
            if mask.size(1) == 1:
                mask = mask.expand(-1, query.size(1), -1)  # (batch, time1, time2)
            mask = unfold_mask(mask, pad_size).unsqueeze(1).eq(0)  # (batch, 1, time1, span)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, span)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, window)

        self.attn = self.span_mask(self.attn)
        self.attn = self.attn / (self.attn.sum(-1, keepdim=True) + 1e-8)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn.unsqueeze(-2), v.transpose(-2, -1)).squeeze(-2)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def get_mean_span(self):
        """a loss term for regularizing the span length"""
        return self.span_mask.span_size.mean() * self.max_span

    def clamp_param(self):
        self.span_mask.clamp_param()


class DynamicSpanMask(nn.Module):
    """Soft masking function for dynamic size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.
    Args:
        max_size: maximum size (i.e. input dimension)
        ramp_size: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """

    def __init__(self, n_feat, n_head, max_span, span_ramp, init_val=0):
        super(DynamicSpanMask, self).__init__()
        self.max_span = max_span
        self.span_ramp = span_ramp
        self.init_val = init_val
        self.dynamic_linear = nn.Linear(n_feat, n_head)
        mask_template = torch.linspace(1 - max_span, 0, steps=max_span)
        self.register_buffer('mask_template', mask_template)
        self.span_size = torch.zeros(1)

    def forward(self, x):
        """mask attention with the right span.

        :param torch.Tensor x: attention scores (batch, head, time1, window_size)
        :param torch.Tensor span: dynamic spans (batch, head, time1)
        :param torch.Tensor/int time2: original key/value sequence length
        """
        mask = self.mask_template + self.span_size.unsqueeze(-1) * self.max_span  # (batch, head, time1, max_span)
        mask = mask / self.span_ramp + 1
        mask = mask.clamp(0, 1)
        if x.size(-1) < self.max_span:
            mask = mask[:, :, -x.size(-1):]
        x = x * mask
        return x

    def get_current_max_size(self, query, include_ramp=True):
        self.span_size = self.get_dynamic_span(query)
        current_size = math.ceil(self.span_size.max().item() * self.max_span)
        if include_ramp:
            current_size += self.span_ramp
        current_size = max(0, min(self.max_span, current_size))
        return current_size

    def get_dynamic_span(self, query):
        """compute the dynamic span."""
        return self.max_span * torch.sigmoid(self.dynamic_linear(query).transpose(1, 2))   # (B, n_head, time)

    def get_current_avg_size(self, include_ramp=True):
        current_size = math.ceil(self.span_size.mean().item() * self.max_span)
        if include_ramp:
            current_size += self.span_ramp
        current_size = max(0, min(self.max_span, current_size))
        return current_size

    def clamp_param(self):
        pass


from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
class MultiHeadedAttentionFixedSpan(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate=0, span_size=100, span_ratio=0.5):
        super(MultiHeadedAttentionFixedSpan, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.span_size = span_size
        self.span_ratio = span_ratio

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time1, time2, size)
        :param torch.Tensor value: (batch, time1, time2, size)
        :param torch.Tensor mask: (batch, 1, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)  # (batch, head, time1, d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)  # (batch, head, time2, d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)  # (batch, head, time2, d_k)

        pad_size = (int(self.span_size*self.span_ratio), self.span_size-int(self.span_size*self.span_ratio)-1)
        new_k = F.pad(k, (0, 0, *pad_size), 'constant', 0).unfold(-2, self.span_size, 1)  # (batch, head, time2, d_k, span)
        new_v = F.pad(v, (0, 0, *pad_size), 'constant', 0).unfold(-2, self.span_size, 1)  # (batch, head, time2, d_k, span)
        del k, v

        scores = (torch.matmul(q.unsqueeze(-2), new_k) / math.sqrt(self.d_k)).squeeze(-2)  # (batch, head, time1, window)
        if mask is not None:
            if mask.size(1) == 1:
                mask = mask.expand(-1, scores.size(2)//mask.size(1), -1)  # (batch, time1, time2)
            mask = unfold_mask(mask, pad_size).unsqueeze(1).eq(0)  # (batch, 1, time1, span)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, window)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, window)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn.unsqueeze(-2), new_v.transpose(-2, -1)).squeeze(-2)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)


class AbsSpanMask(nn.Module):
    """Soft masking function for adaptive size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.
    Args:
        max_span: maximum size (i.e. input dimension)
        span_ramp: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """
    def __init__(self, max_span, max_len, span_ramp, span_ratio, ratio_adaptive, shape, causal_flag):
        super(AbsSpanMask, self).__init__()
        self.max_span = max_span
        self.max_len = max_len
        self.span_ramp = span_ramp
        mask_template = torch.linspace(0, max_len - 1, steps=max_len)
        self.register_buffer('mask_template', mask_template)
        if ratio_adaptive and causal_flag is False:
            self.span_ratio = nn.Parameter(torch.zeros(*shape) + span_ratio)
        else:
            if causal_flag is False:
                span_ratio = torch.zeros(*shape) + span_ratio
            else:
                span_ratio = torch.ones(*shape)
            self.register_buffer('span_ratio', span_ratio)
        self.span_size_clip_range = [0.5, 1]
        self.span_ratio_clip_range = [0.2, 0.8]
        self.causal_flag = causal_flag

    def forward(self, x, query):
        raise NotImplementedError

    def get_current_max_size(self, include_ramp=True):
        raise NotImplementedError

    def get_current_avg_size(self, include_ramp=True):
        raise NotImplementedError

    def generarte_mask(self, time):
        """
        tensor([[-0., -1., -2., -3., -4., -5., -6., -7., -8., -9.],
                [-1., -0., -1., -2., -3., -4., -5., -6., -7., -8.],
                [-2., -1., -0., -1., -2., -3., -4., -5., -6., -7.],
                [-3., -2., -1., -0., -1., -2., -3., -4., -5., -6.],
                [-4., -3., -2., -1., -0., -1., -2., -3., -4., -5.],
                [-5., -4., -3., -2., -1., -0., -1., -2., -3., -4.],
                [-6., -5., -4., -3., -2., -1., -0., -1., -2., -3.],
                [-7., -6., -5., -4., -3., -2., -1., -0., -1., -2.],
                [-8., -7., -6., -5., -4., -3., -2., -1., -0., -1.],
                [-9., -8., -7., -6., -5., -4., -3., -2., -1., -0.]])
        """
        if time > self.max_len:
            self.max_len = time
            self.mask_template = torch.linspace(0, self.max_len - 1, steps=self.max_len, 
                                                device=self.mask_template.device)
        if time < self.max_len:
            mask = self.mask_template[-time:]
        mask = -1 * torch.abs(mask - mask.unsqueeze(1))  # distance from the central frame, (time, time)
        return mask

    def clamp_param(self):
        pass


class AdaptiveSpanMask2(AbsSpanMask):
    """Soft masking function for adaptive size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.
    Args:
        max_span: maximum size (i.e. input dimension)
        span_ramp: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """
    def __init__(self, n_feat, n_head, max_span, span_ramp=2, init_val=1, shape=(1,), span_ratio=0.5, ratio_adaptive=False, max_len=800, causal_flag=False):
        super(AdaptiveSpanMask2, self).__init__(max_span, max_len, span_ramp, span_ratio, ratio_adaptive, shape, causal_flag)
        self.span_size = nn.Parameter(torch.zeros(*shape) + init_val)

    def forward(self, x, query):
        mask = self.generarte_mask(x.size(-1))  # distance from the central frame, (time, time)

        span_size = self.span_size * self.max_span      # (head, 1, 1)
        left_span_size = torch.floor(span_size * self.span_ratio)    # (head, 1, 1)
        right_span_size = span_size - left_span_size - 1  # (head, 1, 1)

        mask = mask + 1 + \
               left_span_size * mask.new_ones(mask.size()).tril() + \
               right_span_size * mask.new_ones(mask.size()).triu(1)   # (head, time, time)
        mask = mask / self.span_ramp + 1   # (head, time, time)

        x = x * mask.clamp(0, 1)
        return x.masked_fill(mask.le(0), 0.0)   # (batch, head, time, time)

    def get_current_max_size(self, include_ramp=True):
        current_size = math.ceil(self.span_size.max().item() * self.max_span)
        if include_ramp:
            current_size += self.span_ramp
        current_size = max(0, min(self.max_span, current_size))
        return current_size

    def get_current_avg_size(self, include_ramp=True):
        current_size = math.ceil(self.span_size.mean().item() * self.max_span)
        if include_ramp:
            current_size += self.span_ramp
        current_size = max(0, min(self.max_span, current_size))
        return current_size

    def clamp_param(self):
        self.span_size.data.clamp_(self.span_size_clip_range[0], self.span_size_clip_range[1])
        if torch.is_tensor(self.span_ratio) and self.causal_flag is False:
            self.span_ratio.data.clamp_(self.span_ratio_clip_range[0], self.span_ratio_clip_range[1])


class FixedSpanMask2(AbsSpanMask):
    """Soft masking function for adaptive size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.
    Args:
        max_span: maximum size (i.e. input dimension)
        span_ramp: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """
    def __init__(self, n_feat, n_head, max_span, span_ramp=2, init_val=1, shape=(1,), span_ratio=0.5, ratio_adaptive=False, max_len=800, causal_flag=False):
        super(FixedSpanMask2, self).__init__(max_span, max_len, span_ramp, span_ratio, ratio_adaptive, shape, causal_flag)
        span_size = torch.ones(*shape)
        self.register_buffer('span_size', span_size)

    def forward(self, x, query):
        mask = self.generarte_mask(x.size(-1))  # distance from the central frame, (time, time)

        span_size = self.span_size * self.max_span  # (head, 1, 1)
        left_span_size = torch.floor(span_size * self.span_ratio)    # (head, 1, 1)
        right_span_size = span_size - left_span_size - 1  # (head, 1, 1)

        mask = mask + 1 + \
               left_span_size * mask.new_ones(mask.size()).tril() + \
               right_span_size * mask.new_ones(mask.size()).triu(1)  # (head, time, time)

        return x.masked_fill(mask.le(0), 0.0)   # (batch, head, time, time)

    def get_current_max_size(self, include_ramp=True):
        return self.span_size * self.max_span

    def get_current_avg_size(self, include_ramp=True):
        return self.span_size * self.max_span

    def clamp_param(self):
        if torch.is_tensor(self.span_ratio) and self.causal_flag is False:
            self.span_ratio.data.clamp_(self.span_ratio_clip_range[0], self.span_ratio_clip_range[1])


class DynamicSpanMask2(AbsSpanMask):
    """Soft masking function for dynamic size.
    It masks out the last K values of an input. The masking value
    goes from 1 to 0 gradually, so K can be learned with
    back-propagation.
    Args:
        max_size: maximum size (i.e. input dimension)
        ramp_size: size of the ramp going from 0 to 1
        init_val: initial size proportion not to be masked out
        shape: learn multiple sizes independent of each other
    """

    def __init__(self, n_feat, n_head, max_span, span_ramp=2, init_val=1, shape=(1,), span_ratio=0.5, ratio_adaptive=False, max_len=800, causal_flag=False):
        super(DynamicSpanMask2, self).__init__(max_span, max_len, span_ramp, span_ratio, ratio_adaptive, shape, causal_flag)
        self.dynamic_linear = nn.Linear(n_feat//n_head, 1)
        span_size = torch.zeros(*shape)
        #span_size = torch.zeros(shape[0])
        self.register_buffer('span_size', span_size)

    def forward(self, x, query):
        """mask attention with the right span.

        :param torch.Tensor x: attention scores (batch, head, time1, window_size)
        :param torch.Tensor query: query (batch, head, time1) (batch, head, time1, d_k)
        :param torch.Tensor/int time2: original key/value sequence length
        """
        mask = self.generarte_mask(x.size(-1))  # distance from the central frame, (time, time)

        span_size = self.get_dynamic_span(query)  # (batch, head, time, 1)
        left_span_size = torch.floor(span_size * self.span_ratio)  # (batch, head, time, 1) x (head, 1, 1)
        right_span_size = span_size - left_span_size - 1   # (batch, head, time, 1) x (head, 1, 1)

        self.span_size = torch.sum(torch.sum(span_size, dim=0).view(*span_size.size()[1:]), dim=1, keepdim=True)

        mask = mask + 1 + \
               left_span_size * mask.new_ones(mask.size()).tril() + \
               right_span_size * mask.new_ones(mask.size()).triu(1)   # (batch, head, time, time)
        mask = mask / self.span_ramp + 1

        x = x * mask.clamp(0, 1)
        return x.masked_fill(mask.le(0), 0.0)    # (batch, head, time, time)

    def get_dynamic_span(self, query):
        """compute the dynamic span."""
        span_size = torch.sigmoid(self.dynamic_linear(query))   # (batch, head, time, 1)
        return self.max_span * span_size.clamp(self.span_size_clip_range[0], self.span_size_clip_range[1])

    def get_current_max_size(self, include_ramp=True):
        current_size = math.ceil(self.span_size.max().item() * self.max_span)
        if include_ramp:
            current_size += self.span_ramp
        current_size = max(0, min(self.max_span, current_size))
        return current_size

    def get_current_avg_size(self, include_ramp=True):
        current_size = math.ceil(self.span_size.mean().item() * self.max_span)
        if include_ramp:
            current_size += self.span_ramp
        current_size = max(0, min(self.max_span, current_size))
        return current_size

    def clamp_param(self):
        if torch.is_tensor(self.span_ratio) and self.causal_flag is False:
            self.span_ratio.data.clamp_(self.span_ratio_clip_range[0], self.span_ratio_clip_range[1])


span_mask = dict(
    adaptive=AdaptiveSpanMask2,
    dynamic=DynamicSpanMask2,
    fixed=FixedSpanMask2,
)
class MultiHeadedAttention2(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """
    def __init__(self, n_head, n_feat, dropout_rate, span_type=None, max_span=100, span_ramp=2, span_init=0, span_ratio=0.5, ratio_adaptive=False, causal_flag=False):
        super(MultiHeadedAttention2, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        if span_type is None:
            self.span_mask = None
        elif span_type in span_mask:
            self.max_span = max_span
            self.span_ratio = span_ratio
            self.span_mask = span_mask[span_type](n_feat=n_feat,
                                                  n_head=n_head,
                                                  max_span=self.max_span,
                                                  span_ramp=span_ramp,
                                                  init_val=span_init,
                                                  shape=(n_head, 1, 1),
                                                  span_ratio=span_ratio,
                                                  ratio_adaptive=ratio_adaptive,
                                                  causal_flag=causal_flag)


    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)   # (batch, head, time1, time2)

        if self.span_mask is not None:
            self.attn = self.span_mask(self.attn, q)
            self.attn = self.attn / (self.attn.sum(3, keepdim=True) + 1e-20)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def get_mean_span(self):
        """a loss term for regularizing the span length"""
        return self.span_mask.span_size.mean() * self.max_span

    def get_mean_ratio(self):
        """a loss term for regularizing the span length"""
        return self.span_mask.span_ratio.mean()

    def clamp_param(self):
        self.span_mask.clamp_param()