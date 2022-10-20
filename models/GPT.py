'''
TODO : 
Layernorm이 모델 전반적으로 적용되기 때문에, weight initialization으로는 N(0, 0.02)을 사용.
변형된 L2 regularization(w=0.01)을 non bias나 gain weights들에 적용하였다
'''

import torch.nn as nn
import torch.nn.functional as F

from model_utils.modules import *
from model_utils.utils import get_transformer_mask


class TransformerBlocks(nn.Module):
    def __init__(self,
            pad_id: int, # padding을 나타내는 id.
            n_layer: int = 6, # 몇 개의 Encoder Layer로 구성할 것인지.
            h_dim: int = 512, # 각각의 레이어는 몇 개의 hidden dimension을 가질 것인지.
            num_heads: int = 8, # multi-head attention에서 몇 개의 head를 쓸 지.
            d_ff: int = 2048, # Position-wise의 중간 hidden dimension.
            dropout: float = 0.1, # dropout rate
        ):
        super(TransformerBlocks, self).__init__()
        self.n_layer = n_layer
        self.num_heads = num_heads
        self.pad_id = pad_id
        
        self.attn = MultiHeadAttention(dim=h_dim, num_heads=num_heads)
        self.pwfc = PositionwiseFFN(input_dim=h_dim, d_ff=d_ff, output_dim=h_dim, activation='gelu')
        self.layerNorm = nn.LayerNorm(h_dim)
        self.encoder_layer = TransformerBlock(self.attn, self.pwfc, self.layerNorm, dropout)
        self.forwardLayerList = nn.ModuleList([self.encoder_layer for _ in range(n_layer)])
    
    def forward(self, inputs, mask):
        x = inputs
        for layer in self.forwardLayerList:
            x = layer(x, mask)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        token_len: int, # 학습하려는 언어의 token dictionary 크기.
        pad_id: int, # padding을 나타내는 id.
        n_layer: int = 12, # 몇 개의 Encoder Layer로 구성할 것인지.
        h_dim: int = 768, # 각각의 레이어는 몇 개의 hidden dimension을 가질 것인지.
        num_heads: int = 12, # multi-head attention에서 몇 개의 head를 쓸 지.
        d_ff: int = 3072, # Position-wise의 중간 hidden dimension.
        dropout: float = 0.1, # dropout rate
        seq_max_len: int = 512, # 생성할 최대 문장 길이.
        contain_last_layer: bool = True, # pre-train에 필요한 마지막 linear layer를 추가할 지.(model load 시에 사용.)
    ):
        super(GPT, self).__init__()
        self.token_len = token_len
        self.pad_id = pad_id

        # initialize layers
        self.masking = get_transformer_mask
        self.word_embedding = nn.Embedding(token_len, h_dim, padding_idx=pad_id)
        self.positional_embedding = nn.Embedding(seq_max_len, h_dim, padding_idx=pad_id)

        self.transformer_blocks = TransformerBlocks(pad_id, n_layer, h_dim, num_heads, d_ff, dropout)

        self.linear = nn.Linear(h_dim, token_len)
        # share weights with embedding layer (W_e)
        self.linear.weight = self.word_embedding.weight

    def forward(self, inputs, positions):
        mask = self.masking(batch=inputs, pad_id=self.pad_id, mask_type="decoder")
        inputs = self.word_embedding(inputs)
        inputs += self.positional_embedding(positions)
        outputs = self.transformer_blocks(inputs, mask)
        if self.contain_last_layer:
            outputs = self.linear(outputs)
            outputs = F.softmax(outputs, dim=-1)

        return outputs


class finetuning_GPT(nn.Module):
    def __init__(
        self,
        model_path: str, # saved pretrained weights path.
        config: dict, # config for pretrained model.
    ):
        super(finetuning_GPT, self).__init__()
        # load pretrained model
        pretrained_state_dict = torch.load(model_path)
        config['contain_last_layer'] = False
        self.pretrained_model = GPT(**config)
        pretrained_state_dict = self.load_weights_without_last_layer(
            pretrained_model.state_dict(), pretrained_state_dict
        )
        self.pretrained_model.load_state_dict(pretrained_state_dict)

        # new linear layer for fine-tuning
        self.linear = nn.Linear(h_dim, n_class)

    def load_weights_without_last_layer(self, model_dict, pretrained_dict):
        # load weights without last linear layer.
        # 1. filter out unnecessary keys
        pretrained_weights = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_weights) 

        return model_dict


    def forward(self, inputs, positions):
        transformer_outputs = self.pretrained_model(inputs, positions)
        outputs = self.linear(transformer_outputs)
        outputs = F.softmax(outputs, dim=-1)

        return outputs
