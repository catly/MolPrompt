# import torchvision.models as models
import torch
import torch.nn as nn
# from graphormer_v1 import Graphormer
from transformers import GraphormerModel

class GraphEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(GraphEncoder, self).__init__()
        if pretrained:  # if use pretrained scibert model

            self.graph_model = GraphormerModel.from_pretrained(
                'all_checkpoints/graphormer_pretrained/graphormer-base-pcqm4mv1')

        # self.dropout = nn.Dropout(0.1)
        self.graph_encoder = self.graph_model.graph_encoder

        self.graph_node_feature = self.graph_encoder.graph_node_feature
        self.droput_module = self.graph_encoder.dropout_module
        self.graph_attn_bias = self.graph_encoder.graph_attn_bias

        self.layers = self.graph_encoder.layers
        # self.hidden_size = self.main_model.config.hidden_size

    def forward(self, x, attn_bias, attn_edge_type, spatial_pos,
                        in_degree, out_degree, edge_input, prompt_emb):

        device = torch.device(f'cuda:1')
        self.graph_model = self.graph_model.to(device)

        input_nodes = x
        perturb = None
        inner_states, graph_rep = self.graph_encoder(input_nodes, edge_input, attn_bias, in_degree, out_degree,
                                    spatial_pos, attn_edge_type, perturb=perturb)

        data_x = input_nodes
        n_graph, n_node = data_x.size()[:2]
        padding_mask = (data_x[:, :, 0]).eq(0)
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        attn_bias = self.graph_attn_bias(input_nodes, attn_bias, spatial_pos, edge_input, attn_edge_type)

        inner = inner_states[0]
        inner_re = inner.clone()
        prompt_emb = prompt_emb[:, 0]
        inner_re[-1] = prompt_emb
        inner_list = []
        for layer in self.layers:
            inner_re, _ = layer(
                input_nodes=inner_re,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=None,
                self_attn_bias=attn_bias,
            )
            inner_list.append(inner_re)
        graph_rep_out = inner_re[0, :, :]


        return graph_rep_out


if __name__ == '__main__':
    model = GraphEncoder()
    for name, param in model.named_parameters():
        print(name)
