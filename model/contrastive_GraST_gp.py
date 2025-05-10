import torch
import torch.nn as nn
import torch.nn.functional as F

from model.bert import TextEncoder
from model.graphormer_0313 import GraphEncoder
from model.prompt_encoder import PromptEncoder

import pytorch_lightning as pl
from torch import optim

from config import parse_args


class GraSTSimclr(pl.LightningModule):
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

        # Graph Encoder
        self.graph_encoder = GraphEncoder(pretrained=True)
        # for name, param in self.graph_encoder.named_parameters():
        #     print(name)

        # Text Encoder
        self.text_encoder = TextEncoder(pretrained=True)
        # LiuC: 考虑是否不导入kvplm_pretrained
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

    def forward(self, features_graph, features_text):

        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        logits_per_graph = features_graph @ features_text.t() / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        return logits_per_graph, logits_per_text, loss


    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch_idx, batch):
        device = torch.device(f'cuda:1')

        attn_bias = batch.attn_bias
        attn_edge_type = batch.attn_edge_type
        spatial_pos = batch.spatial_pos
        in_degree = batch.in_degree
        out_degree = batch.out_degree
        x = batch.x
        edge_input = batch.edge_input
        # text
        text = batch.text
        text_mask = batch.text_mask
        text_batch = torch.cat(text, dim=0)
        text_mask_batch = torch.cat(text_mask, dim=0)
        # prompt
        prompt_text = batch.prompt_text
        prompt_mask = batch.prompt_mask
        prompt_text_batch = torch.cat(prompt_text, dim=0)
        prompt_mask_batch = torch.cat(prompt_mask, dim=0)


        # LiuC: to(device) cuda:0
        # graph
        attn_bias = attn_bias.to(device)
        attn_edge_type = attn_edge_type.to(device)
        spatial_pos = spatial_pos.to(device)
        in_degree = in_degree.to(device)
        out_degree = out_degree.to(device)
        x = x.to(device)
        edge_input = edge_input.to(device)
        # text
        text_batch = text_batch.to(device)
        text_mask_batch = text_mask_batch.to(device)
        # prompt
        prompt_text_batch = prompt_text_batch.to(device)
        prompt_mask_batch = prompt_mask_batch.to(device)
        prompt_emb, prompt_extended_mask = self.prompt_encoder(prompt_text_batch, prompt_mask_batch)


        text_feature = self.text_encoder(text_batch, text_mask_batch)
        graph_feature = self.graph_encoder(x, attn_bias, attn_edge_type, spatial_pos,
                                      in_degree, out_degree, edge_input, prompt_emb)

        # LiuC: proj: 768->256
        graph_feature = self.graph_proj_head(graph_feature)
        text_feature = self.text_proj_head(text_feature)

        # LiuC: self.forward()
        _, _, loss = self.forward(graph_feature, text_feature)

        self.log("train_loss", loss)
        # return loss
        return text_feature, graph_feature, loss



if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = torch.device(f'cuda:0')

    graph_encoder = GraphEncoder(pretrained=True)
    for name, param in graph_encoder.named_parameters():
        print(name)

    print('##############')
    text_encoder = TextEncoder(pretrained=True)
    text_ckpt = torch.load('../all_checkpoints/kvplm_pretrained/ckpt_KV_1.pt')
    if 'module.ptmodel.bert.embeddings.word_embeddings.weight' in text_ckpt:
        pretrained_dict = {"main_model." + k[20:]: v for k, v in text_ckpt.items()}
    elif 'bert.embeddings.word_embeddings.weight' in text_ckpt:
        pretrained_dict = {"main_model." + k[5:]: v for k, v in text_ckpt.items()}
    else:
        pretrained_dict = {"main_model." + k[12:]: v for k, v in text_ckpt.items()}

    text_encoder.load_state_dict(pretrained_dict, strict=False)
    for name, param in text_encoder.named_parameters():
        print(name)