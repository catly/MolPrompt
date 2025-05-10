# import torchvision.models as models
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import argparse
from config import parse_args

class PromptEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(PromptEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model

            self.main_model = BertModel.from_pretrained(
                        'all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased')

        # prompt_embedding
        self.embed = self.main_model.embeddings
        self.dropout = nn.Dropout(0.1)


    def forward(self, input_ids, attention_mask):
        # device = input_ids.device
        device = torch.device(f'cuda:1')

        self.main_model = self.main_model.to(device)


        typ = torch.zeros(input_ids.shape).long().to(device)

        prompt_emb = self.embed(input_ids, token_type_ids=typ)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        prompt_extended_mask = (1.0 - extended_attention_mask) * -10000.0

        return prompt_emb, prompt_extended_mask