# import torchvision.models as models
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import argparse
from config import parse_args

class TextEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(TextEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model
            # LiuC: 第一种导入方式
            # self.main_model = BertModel.from_pretrained('all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased') \
            self.main_model = BertModel.from_pretrained(
                        'all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased')

            # LiuC: 第二种导入方式
            # bertconfig = BertConfig.from_pretrained('../all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased/config.json')
            # self.main_model = BertModel(bertconfig)

        self.embeddings = self.main_model.embeddings
        self.encoder = self.main_model.encoder
        self.pooler = self.main_model.pooler

        self.dropout = nn.Dropout(0.1)

        # self.hidden_size = self.main_model.config.hidden_size

    # def forward(self, input_ids, attention_mask, prompt):
    def forward(self, input_ids, attention_mask):
        # device = input_ids.device
        device = torch.device(f'cuda:1')
        # pooler
        # self.dense = nn.Linear(768, 768)
        # self.activation = nn.Tanh()

        self.main_model = self.main_model.to(device)
        typ = torch.zeros(input_ids.shape).long().to(device)

        hidden_states_0 = self.embeddings(input_ids, token_type_ids=typ)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        text_extended_mask = (1.0 - extended_attention_mask) * -10000.0

        output = self.encoder(hidden_states_0, text_extended_mask)[0]
        output = self.pooler(output)


        return output


if __name__ == '__main__':
    model = TextEncoder()
    for name, param in model.named_parameters():
        print(name)

