'''
TODO : get output of decoder attention to visualize attention matrix.
'''

import torch.nn as nn
import torch.nn.functional as F

from model_utils.modules import *
from model_utils.utils import get_transformer_mask
from model_utils.embeddings import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(self,
                 token_len: int, # 학습하려는 언어의 token dictionary 크기.
                 pad_id: int, # padding을 나타내는 id.
                 n_layer: int = 6, # 몇 개의 Encoder Layer로 구성할 것인지.
                 h_dim: int = 512, # 각각의 레이어는 몇 개의 hidden dimension을 가질 것인지.
                 num_heads: int = 8, # multi-head attention에서 몇 개의 head를 쓸 지.
                 d_ff: int = 2048, # Position-wise의 중간 hidden dimension.
                 dropout: float = 0.1 # dropout rate
                ):
        super(TransformerEncoder, self).__init__()
        self.n_layer = n_layer
        self.num_heads = num_heads
        self.pad_id = pad_id
        
        self.masking = get_transformer_mask
        self.attn = MultiHeadAttention(dim=h_dim, num_heads=num_heads)
        self.pwfc = PositionwiseFFN(input_dim=h_dim, d_ff=d_ff, output_dim=h_dim)
        self.layerNorm = nn.LayerNorm(h_dim)
        self.encoder_layer = TransformerBlock(self.attn, self.pwfc, self.layerNorm, dropout)
        self.forwardLayerList = nn.ModuleList([self.encoder_layer for _ in range(n_layer)])
        
        self.embedding = nn.Embedding(token_len, h_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding()
    
    def forward(self, inputs):
        mask = self.masking(batch=inputs, pad_id=self.pad_id, mask_type="encoder")
        
        inputs = self.embedding(inputs)
        inputs += self.positional_encoding(inputs.size(1))
        x = inputs
        for layer in self.forwardLayerList:
            x = layer(x, mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,
                 token_len: int, # 학습하려는 언어의 token dictionary 크기.
                 pad_id: int, # padding을 나타내는 id.
                 n_layer: int = 6, # 몇 개의 Encoder Layer로 구성할 것인지.
                 h_dim: int = 512, # 각각의 레이어는 몇 개의 hidden dimension을 가질 것인지.
                 num_heads: int = 8, # multi-head attention에서 몇 개의 head를 쓸 지.
                 d_ff: int = 2048, # Position-wise의 중간 hidden dimension.
                 dropout: float = 0.1 # dropout
                ):
        super(TransformerDecoder, self).__init__()
        self.n_layer = n_layer
        self.num_heads = num_heads
        self.masking = get_transformer_mask
        self.pad_id = pad_id
        
        self.attn = MultiHeadAttention(dim=h_dim, num_heads=num_heads)
        self.pwfc = PositionwiseFFN(input_dim=h_dim, d_ff=d_ff, output_dim=h_dim)
        self.layerNorm = nn.LayerNorm(h_dim)
        self.decoder_layer = TransformerDecoderLayer(self.attn, self.pwfc, self.layerNorm, dropout=dropout)
        self.forwardLayerList = nn.ModuleList([self.decoder_layer for _ in range(self.n_layer)])
        
        self.embedding = nn.Embedding(token_len, h_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding()
    
    def forward(self, inputs, encoder_outputs):
        # 인코더와의 차이점 - 마스킹
        mask = self.masking(batch=inputs, pad_id=self.pad_id, mask_type="decoder")
        
        inputs = self.embedding(inputs)
        inputs += self.positional_encoding(inputs.size(1))
        
        x = inputs
        for layer in self.forwardLayerList:
            x = layer(x, mask, encoder_outputs)
        return x

class Transformers(nn.Module):
    def __init__(self, token_len: int, # 학습하려는 언어의 token dictionary 크기. 
                 pad_id: int,  # padding을 나타내는 id.
                 n_encoder_layer: int = 6, # 몇 개의 Encoder Layer로 구성할 것인지.
                 n_decoder_layer: int = 6, # 몇 개의 Decoder Layer로 구성할 것인지.
                 h_dim: int = 512, # 각각의 레이어는 몇 개의 hidden dimension을 가질 것인지.
                 num_heads: int = 8, # multi-head attention에서 몇 개의 head를 쓸 지.
                 d_ff: int = 2048, # Position-wise의 중간 hidden dimension.
                 dropout: float = 0.1,
                ):
        super(Transformers, self).__init__()
        self.encoder = TransformerEncoder(
            token_len=token_len, 
            pad_id=pad_id, 
            n_layer=n_encoder_layer,
            h_dim=h_dim,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            token_len=token_len, 
            pad_id=pad_id, 
            n_layer=n_decoder_layer,
            h_dim=h_dim,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.linear = nn.Linear(h_dim, token_len)
        
    # inputs와 targets를 따로 받는 이유는 Transformer는 번역에 사용되는 모델이기 때문.
    def forward(self, inputs, targets):
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(targets, encoder_outputs)
        x = self.linear(decoder_outputs)
        x = F.softmax(x, dim=-1)
        return x
