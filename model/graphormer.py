# import torchvision.models as models
import torch
import torch.nn as nn
# from graphormer_v1 import Graphormer
from transformers import GraphormerModel

class GraphEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(GraphEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model

            self.graph_model = GraphormerModel.from_pretrained('all_checkpoints/graphormer_pretrained/graphormer-base-pcqm4mv1')

        self.dropout = nn.Dropout(0.1)
        self.graph_encoder = self.graph_model.graph_encoder
        # self.hidden_size = self.main_model.config.hidden_size

    def forward(self, x, attn_bias, attn_edge_type, spatial_pos,
                                          in_degree, out_degree, edge_input):

        device = torch.device(f'cuda:1')
        self.graph_model = self.graph_model.to(device)

        input_nodes = x
        perturb = None
        # typ = torch.zeros(input_ids.shape).long().to(device)
        # output = self.main_model(input_ids, token_type_ids=typ, attention_mask=attention_mask)['pooler_output']  # b,d
        inner_states, graph_rep = self.graph_encoder(input_nodes, edge_input, attn_bias, in_degree, out_degree,
                                    spatial_pos, attn_edge_type, perturb=perturb)

        return graph_rep


if __name__ == '__main__':
    model = GraphEncoder()
    for name, param in model.named_parameters():
        print(name)
