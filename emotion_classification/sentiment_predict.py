import torch
from emotion_classification import bert

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def deal_message(message, config):
    token = config.tokenizer.tokenize(message)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_id = config.tokenizer.convert_tokens_to_ids(token)
    pad_size = 99
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_id) + [0] * (pad_size - len(token))
            token_id += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_id = token_id[:pad_size]
            seq_len = pad_size
    token_id = torch.LongTensor([token_id]).to(config.device)
    seq_len = torch.LongTensor([seq_len]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)
    return token_id, seq_len, mask


def sentiment(message):
    path = './emotion_classification/data'
    config = bert.Config(path)
    model = bert.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    context = deal_message(message, config)

    res = model(context)
    a = res.size()
    res_dict = {0: 'negative', 1: 'positive'}
    result_index = int(torch.argmax(res, 1)[0])
    result_score = float(torch.max(res.data, 1)[0])
    label = res_dict[result_index]
    output = {'intent': label, 'confidence': result_score}
    return output
