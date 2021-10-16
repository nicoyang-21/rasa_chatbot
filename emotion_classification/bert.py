import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Config(object):
    """配置参数"""

    def __init__(self, path):
        self.dataset_path = path + '/weibo_senti_100k.csv'
        self.class_list = [x.strip() for x in open(path + '/class.txt').readlines()]
        self.save_path = path + '/bert.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 128
        self.pad_size = 99  # 添加['cls']
        self.learning_rate = 2e-5
        self.bert_path = './emotion_classification/bert_model'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, return_dict=False)
        output = self.fc(pooled)
        return self.softmax(output)






