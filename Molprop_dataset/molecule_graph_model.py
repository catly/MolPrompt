import torch
import torch.nn as nn
import torch.nn.functional as F


from model.bert import TextEncoder
from model.graphormer_0313 import GraphEncoder
from model.prompt_encoder import PromptEncoder

import pytorch_lightning as pl
from torch import optim



class Graph_pred(pl.LightningModule):
    def __init__(
            self,
            temperature,
            drop_ratio,
            graph_hidden_dim,
            bert_hidden_dim,
            # bert_pretrain,
            projection_dim,
            lr,
            weight_decay,
            num_tasks
    ):
        super().__init__()
        self.save_hyperparameters()

        self.temperature = temperature
        self.drop_ratio = drop_ratio
        self.graph_hidden_dim = graph_hidden_dim
        self.bert_hidden_dim = bert_hidden_dim
        # self.bert_pretrain = bert_pretrain
        self.projection_dim = projection_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_tasks = num_tasks

        # Graph Encoder
        self.graph_encoder = GraphEncoder(pretrained=True)

        # Text Encoder
        self.text_encoder = TextEncoder(pretrained=True)

        text_ckpt = torch.load('all_checkpoints/kvplm_pretrained/ckpt_KV_1.pt')
        if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in text_ckpt:
            pretrained_dict = {"main_model." + k[20:]: v for k, v in text_ckpt.items()}
        elif 'bert.embeddings.word_embeddings.weight' in text_ckpt:
            pretrained_dict = {"main_model." + k[5:]: v for k, v in text_ckpt.items()}
        else:
            pretrained_dict = {"main_model." + k[12:]: v for k, v in text_ckpt.items()}

        self.text_encoder.load_state_dict(pretrained_dict, strict=False)

        self.prompt_encoder = PromptEncoder(pretrained=True)
        prompt_ckpt = torch.load('all_checkpoints/kvplm_pretrained/ckpt_KV_1.pt')
        if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in prompt_ckpt:
            pretrained_dict = {"main_model." + k[20:]: v for k, v in prompt_ckpt.items()}
        elif 'bert.embeddings.word_embeddings.weight' in prompt_ckpt:
            pretrained_dict = {"main_model." + k[5:]: v for k, v in prompt_ckpt.items()}
        else:
            pretrained_dict = {"main_model." + k[12:]: v for k, v in prompt_ckpt.items()}

        self.prompt_encoder.load_state_dict(pretrained_dict, strict=False)

        self.graph_proj_head = nn.Sequential(
            nn.Linear(self.graph_hidden_dim, self.graph_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.graph_hidden_dim, self.projection_dim)
        )
        self.text_proj_head = nn.Sequential(
            nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.bert_hidden_dim, self.projection_dim)
        )


        self.mult = 1
        self.graph_pred_linear = nn.Linear(self.mult * self.graph_hidden_dim, self.num_tasks)


    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


    def forward(self, batch):
        device = torch.device(f'cuda:1')

        attn_bias = batch.attn_bias
        attn_edge_type = batch.attn_edge_type
        spatial_pos = batch.spatial_pos
        in_degree = batch.in_degree
        out_degree = batch.out_degree
        x = batch.x
        edge_input = batch.edge_input

        prompt_text = batch.prompt_text
        prompt_mask = batch.prompt_mask
        prompt_text_batch = torch.cat(prompt_text, dim=0)
        prompt_mask_batch = torch.cat(prompt_mask, dim=0)


        # graph
        attn_bias = attn_bias.to(device)
        attn_edge_type = attn_edge_type.to(device)
        spatial_pos = spatial_pos.to(device)
        in_degree = in_degree.to(device)
        out_degree = out_degree.to(device)
        x = x.to(device)
        edge_input = edge_input.to(device)

        prompt_text_batch = prompt_text_batch.to(device)
        prompt_mask_batch = prompt_mask_batch.to(device)
        prompt_emb, prompt_extended_mask = self.prompt_encoder(prompt_text_batch, prompt_mask_batch)

        graph_feature = self.graph_encoder(x, attn_bias, attn_edge_type, spatial_pos,
                                           in_degree, out_degree, edge_input, prompt_emb)

        output = self.graph_pred_linear(graph_feature)


        return graph_feature, output
