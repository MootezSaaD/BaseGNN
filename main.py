import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import GGNN
from trainer import train
from utils.utils import tally_param, debug


if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='ggnn')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')

    parser.add_argument('--data_src', type=str, help='CSV file of the dataset.', required=True)
    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=128)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=32)
    parser.add_argument('--read_out', type=str, help='GNN readout function', default=32, choices=['sum', 'mean'], default='sum')
    parser.add_argument('--emb_type', type=str, help='Embedding method for node feature generation. Wor2Vec or Transformer-based', choices=['w2v', 'hf'], default='w2v')
    parser.add_argument('--w2v', type=str, help='Pretrained Word2Vec model path, when w2v is selected as an embedding method.')
    parser.add_argument('--tok', type=str, help='Tokenization method of the src code.', choices=['ntlk', 'hf'], default='ntlk')
    parser.add_argument('--build_method', type=str, help='Graph construction method. UTC or IFC.', choices=['utc', 'ifc'], default='itc')
    args = parser.parse_args()

    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataset = DataSet(args)

    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'

    model = GGNN(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type, read_out=args.read_out)

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    model.cuda()
    loss_function = BCELoss(reduction='sum')
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    train(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
          loss_function=loss_function, optimizer=optim,
          save_path=model_dir + '/GGNN', max_patience=50, log_every=None)
