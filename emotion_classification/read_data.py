import time
from datetime import timedelta

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def data_split(path):
    data = pd.read_csv(path)
    for i, j in enumerate(data['label']):
        if j in [1, 2, 3]:
            data['label'][i] = 1
    content = data['review']
    label = data['label']
    train_content, test_content_t, train_label, test_label_t = train_test_split(content, label, test_size=0.2,
                                                                                random_state=100)
    test_t_len = len(test_label_t)
    test_len = test_t_len // 2
    test_content = test_content_t[:test_len]
    test_label = test_label_t[:test_len]
    eval_content = test_content_t[test_len:]
    eval_label = test_label_t[test_len:]
    return train_content, train_label, test_content, test_label, eval_content, eval_label


def build_dataset(config):
    def load_dataset(content, label, pad_size=config.pad_size):
        contents = []
        for line, label in tqdm(zip(content, label)):
            line = line.strip()
            token = config.tokenizer.tokenize(line)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_id = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_id) + [0] * (pad_size - len(token))
                    token_id += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_id = token_id[:pad_size]
                    seq_len = pad_size
            contents.append((token_id, int(label), seq_len, mask))
        return contents

    train_content, train_label, test_content, test_label, eval_content, eval_label = data_split(config.dataset_path)
    train = load_dataset(train_content, train_label)
    test = load_dataset(test_content, test_label)
    evaluation = load_dataset(eval_content, eval_label)
    return train, test, evaluation


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, data):
        x = torch.LongTensor([_[0] for _ in data]).to(self.device)
        y = torch.LongTensor([_[1] for _ in data]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in data]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in data]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
