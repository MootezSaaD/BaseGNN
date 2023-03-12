import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f
import dgl

class GGNN(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, read_out, num_steps=8):
        super(GGNN, self).__init__()
        self.read_out = read_out
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, graph, cuda=False):

        node_features = graph.ndata['features']
        edges = graph.edata['etype']

        node_num = graph.number_of_nodes()
        zero_pad = torch.zeros(
            [node_num, self.out_dim - self.inp_dim],
            dtype=torch.float,
            device=node_features.device,
        )
        h1 = torch.cat([node_features, zero_pad], -1)
        out = self.ggnn(graph, h1, edges)
        out = torch.cat([out, node_features], -1)

        graph.ndata['h'] = out

        if self.read_out == 'sum':
            feats = dgl.sum_nodes(graph, 'h')
        if self.read_out == 'mean':
            feats = dgl.mean_nodes(graph, 'h')

        result = self.sigmoid(self.classifier(feats))
        
        return result