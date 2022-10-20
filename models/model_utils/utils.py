'''
언어 모델 구현에 필요한 유틸리티들.
'''

import torch
    

def get_transformer_mask(batch: torch.Tensor, pad_id: int, mask_type: str):
    '''
    Transformer에 맞는 input 형태로 마스킹 해주는 함수.
    
    batch : 모델의 input batch.
    pad_id : pad_token의 id
    mask_type : transformer 인코더의 input인지 디코더의 input인지 ["encoder", "decoder"]로 명시.
    
    encoder mask output : [batch_size, max_sequence_length] 크기의 0과 1로 이루어진 mask.
    decoder mask  output : [batch_size, max_sequence_length, max_sequence_length] 크기의 0과 1로 이루어진 mask.
    mask 부분은 False, 나머진 True.
    '''
    assert mask_type in ["encoder", "decoder"], "mask_type should be 'encoder' or 'decoder'!"
    padding_mask = (batch != pad_id).unsqueeze(1) # (B, 1, L)
    max_len = batch.shape[1]
    if mask_type.lower() == "encoder":
        return padding_mask.repeat(1, max_len, 1)
    nopeak_mask = torch.ones([1, max_len, max_len], dtype=torch.bool)  # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask)  # (1, L, L)
    mask = padding_mask & nopeak_mask # (B, L, L)
    
    return mask

