import dgl, torch

class DataEntry:
    def __init__(self, datset, adj_matrix, features, target):
        self.dataset = datset
        self.target = target
        self.graph = dgl.from_scipy(adj_matrix)
        self.features = torch.FloatTensor(features)
        self.graph.ndata['features'] = self.features
        self.graph.edata['etype'] = torch.ones(self.graph.num_edges(), dtype=torch.int32)