import copy
import json
import sys
import csv
import os

import torch
from dgl import DGLGraph
import dgl
from tqdm import tqdm
from utils.graph_construction import utc, ifc
from utils.init_features import load_wv, nltk_tokenizer
from utils.io import read_file

from data_loader.batch_graph import GGNNBatchGraph
from utils.utils import load_default_identifiers, initialize_batch, debug


class DataEntry:
    def __init__(self, datset, adj_matrix, features, target):
        self.dataset = datset
        self.target = target
        self.graph = dgl.from_scipy(adj_matrix)
        self.features = torch.FloatTensor(features)
        self.graph.ndata['features'] = self.features
        self.graph.edata['etype'] = torch.ones(self.graph.num_edges(), dtype=torch.int32)


class DataSet:
    def __init__(self, args):
        self.args = args
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = self.args.batch_size
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.read_dataset()
        self.initialize_dataset()

    def initialize_dataset(self):
        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self):
        debug('Reading Train File!')
        with open(self.args.data_src) as fp:
            _data = list(csv.DictReader(fp, delimiter=','))[1:] # Skip header
            for entry in tqdm(_data):
                code = read_file(entry['file'])
                label = entry['label']
                if self.args.emb_type == 'w2v':
                    embeddings = load_wv(self.args.w2v)
                if self.args.tok == 'nltk':
                    tokens = nltk_tokenizer(code)
                if self.args.build_method == 'ifc':
                    adj, features = ifc(tokens, embeddings, self.args)
                if self.args.build_method == 'utc':
                    adj, features = utc(tokens, embeddings, self.args)

                example = DataEntry(datset=self, adj_matrix=adj, features=features, target=label)
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)
                    debug('Feature Size %d' % self.feature_size)
                if entry['split'] == 'train':
                    self.train_examples.append(example)
                elif entry['split'] == 'test':
                    self.test_examples.append(example)
                elif entry['split'] == 'val':
                    self.valid_examples.append(example)
                else:
                    raise ValueError('Incorrect split member!!!')


    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=True)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size)
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        taken_entries = [entries[i] for i in ids]
        labels = [e.target for e in taken_entries]
        batch_graph = GGNNBatchGraph()
        for entry in taken_entries:
            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        return batch_graph, torch.FloatTensor(labels)

    def get_next_train_batch(self):
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        ids = self.train_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)
