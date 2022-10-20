def padding(data):
    '''
    길이가 맞지 않는 문장들에 pad_id를 추가하여 길이를 맞춰준다.
    '''
    max_len = len(max(data, key=len))

    for i, seq in enumerate(data):
        if len(seq) < max_len:
            data[i] = seq + [pad_id] * (max_len - len(seq))
    return data, max_len
