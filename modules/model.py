import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f

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

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = batch.de_batchify_graphs(outputs)
        if self.read_out == 'sum':
            ggnn_ = self.classifier(h_i.sum(dim=1))
        if self.read_out == 'mean':
            ggnn_ = self.classifier(h_i.mean(dim=1))
        result = self.sigmoid(ggnn_).squeeze(dim=-1)
        return result