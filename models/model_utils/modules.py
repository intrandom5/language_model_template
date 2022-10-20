import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
import numpy as np


class DotProductAttention(nn.Module):
    '''
    DotProduct Attention 구현체.
    d_k : key의 dimension.
    '''
    def __init__(self, d_k):
        super(DotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, query, key, value, mask):
        x = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            x.masked_fill_(mask==False, -1e4)
        x /= np.sqrt(self.d_k)
        x = F.softmax(x, dim=-1)
        x = torch.matmul(x, value)
        return x

    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "hidden_dim % num_heads should be zero."
        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, self.d_head * num_heads)
        self.key_proj = nn.Linear(dim, self.d_head * num_heads)
        self.value_proj = nn.Linear(dim, self.d_head * num_heads)
        self.scaled_dot_attn = DotProductAttention(dim)

    def forward(self, query, key, value, mask: Tensor = None):
        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        if mask is not None: # 마스킹.
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        context = self.scaled_dot_attn(query, key, value, mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_head)

        return context
    
class PositionwiseFFN(nn.Module):
    def __init__(self, input_dim=512, d_ff=2048, output_dim=512, activation='relu'):
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(input_dim, d_ff)
        if acitvation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_ff, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    
class TransformerBlock(nn.Module):
    '''
    Transformer의 인코더 sub-layer 구현체.

    attn : attention module.
    pwfc : PositionwiseFFN module.
    layerNorm : layer normalization module.
    dropout : dropout rate.
    '''
    def __init__(self, attn, pwfc, layerNorm, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = attn
        self.pwfc = pwfc
        self.layerNorm = layerNorm
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        residual = x
        x = self.attn(x, x, x, mask)
        x += residual
        x = self.layerNorm(x)

        residual = x
        x = self.pwfc(x)
        x += residual
        x = self.layerNorm(x)
        x = self.dropout(x)
        return x

class TransformerDecoderLayer(nn.Module):
    '''
    Transformer의 디코더 sub-layer 구현체.

    attn : attention module.
    pwfc : PositionwiseFFN module.
    layerNorm : layer normalization module.
    dropout : dropout rate.
    '''
    def __init__(self, attn, pwfc, layerNorm, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.attn = attn
        self.pwfc = pwfc
        self.layerNorm = layerNorm
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask, encoder_outputs):
        residual = x
        x = self.attn(x, x, x, mask)
        x += residual
        x = self.layerNorm(x)
        
        # 인코더와의 차이점
        residual = x
        x = self.attn(x, encoder_outputs, encoder_outputs, mask)
        x += residual
        x = self.layerNorm(x)

        residual = x
        x = self.pwfc(x)
        x += residual
        x = self.layerNorm(x)
        x = self.dropout(x)
        return x
