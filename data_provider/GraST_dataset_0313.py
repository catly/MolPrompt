import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from transformers import BertTokenizer
from transformers import RobertaTokenizer, RobertaModel

from config import parse_args
import re
from torch.utils.data import RandomSampler
import torch_geometric

from data_provider.wrapper import preprocess_item
from data_provider.collator_0313 import collator, Batch
from functools import partial


class GraSTPretrain_STM(Dataset):
    def __init__(self, args, ids, scaf):
        # super(GraSTMatchDataset, self).__init__()
        super(GraSTPretrain_STM, self).__init__()
        self.ids = ids
        self.scaf = scaf
        self.graph_aug = args.graph_aug
        self.text_max_len = args.text_max_len
        self.prompt_max_len = args.prompt_max_len
        self.tokenizer = BertTokenizer.from_pretrained('all_checkpoints/bert_pretrained/allenai/scibert_scivocab_uncased/')
        # self.data_type = args.data_type

    def __len__(self):
        return len(self.scaf)

    def __getitem__(self, index):
        idx = self.scaf[index]
        graph_path = 'preprocessing/PubChem/PubChem_process/graph_PubChem/graph' + '_' + f"{self.ids[idx]}.pt"
        text_path = 'preprocessing/PubChem/PubChem_process/text_PubChem/text' + '_' + f"{self.ids[idx]}.txt"
        prompt_path = 'preprocessing/PubChem/PubChem_process/prompt_PubChem/prompt' + '_' + f"{self.ids[idx]}.txt"

        # load graph data
        graph = torch.load(graph_path)

        data_graph_text = preprocess_item(graph)
        data_graph_text.idx = id

        # load text data
        text_list = []
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text_list.append(line)
            if count > 1:
                break

        # load prompt_text data
        prompt_text_list = []
        count_prompt = 0
        for line in open(prompt_path, 'r', encoding='utf-8'):
            count_prompt += 1
            line.strip('\n')
            prompt_text_list.append(line)
            if count > 1:
                break

        text = mask = None
        prompt_text = prompt_mask = None

        text, mask = self.tokenizer_text(text_list[0])
        prompt_text, prompt_mask = self.tokenizer_prompt(prompt_text_list[0])

        # LiuC:
        data_graph_text.text = text
        data_graph_text.text_mask = mask
        data_graph_text.prompt_text = prompt_text
        data_graph_text.prompt_mask = prompt_mask

        return data_graph_text



    def tokenizer_text(self, text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.text_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask

    def tokenizer_prompt(self, prompt_text):
        tokenizer = self.tokenizer
        sentence_token = tokenizer(text=prompt_text,
                                   truncation=True,
                                   padding='max_length',
                                   add_special_tokens=False,
                                   max_length=self.prompt_max_len,
                                   return_tensors='pt',
                                   return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask














